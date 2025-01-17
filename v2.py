import pandas as pd
import numpy as np
import re
from collections import defaultdict
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime

# CUDA 설정: cuda:1 or cuda:0 ################################################################################################################################################################
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 데이터 로드 ################################################################################################################################################################################
response_path = '/home/leezoo/home/model_v2_dataset/test1(under1)/tcga_response_822(undersample1).csv'
pathway_relation_file = '/home/leezoo/home/Reactom_datasets/reactome/ReactomePathwaysRelation.txt'
gmt_file_path = '/home/leezoo/home/Reactom_datasets/reactome/ReactomePathways.gmt'
gene_list_file_path = '/home/leezoo/home/model_v2_dataset/6640_genes.csv'  
response_df = pd.read_csv(response_path) # 반응 데이터 


# 유전자 목록 ################################################################################################################################################################################
gene_ids = pd.read_csv(gene_list_file_path)['genes'].tolist()
print("gene count:", len(gene_ids)) 
num_genes = len(gene_ids)

# 도메인 레이블 정의
DOMAIN_LABELS = {
    'TCGA': 0,
    'Liu': 1,
    'Hugo': 2,
    'VanAllen': 3,
    'Snyder': 4
}

# Gene-to-Pathway Matrix  ##################################################################################################################################################################
unique_pathways = set()  
gene_to_pathway_data = []
with open(gmt_file_path, 'r') as gmt_file:
    for line in gmt_file:
        fields = line.strip().split('\t')
        pathway_id = fields[1]
        genes = fields[2:]
        unique_pathways.add(pathway_id)
        for gene in genes:
            gene_to_pathway_data.append((gene, pathway_id))

unique_pathways = list(unique_pathways)
gene_to_pathway_matrix = pd.DataFrame(0, index=gene_ids, columns=unique_pathways)

for gene, pathway_id in gene_to_pathway_data:
    if gene in gene_to_pathway_matrix.index:
        gene_to_pathway_matrix.loc[gene, pathway_id] = 1

with open(pathway_relation_file, 'r') as file:
    lines = file.readlines()

pathway_dict = defaultdict(list)
for line in lines:
    parts = re.split(r'\s+|->', line.strip())
    if len(parts) == 2:
        source = parts[0]
        target = parts[1]
        pathway_dict[source].append(target)

pathway_index = {pathway: idx for idx, pathway in enumerate(unique_pathways)}




################################################################################################################################################

# TCGA 데이터 로드
X_gene_tcga = np.load('/home/leezoo/home/model_v2_dataset/test1(under1)/X_gene_data(TCGA).npy')  # 예: (964, 16856, 3)
X_clinical_tcga = np.load('/home/leezoo/home/model_v2_dataset/test1(under1)/X_clinical_data(TCGA).npy')  # 예: (964, 38)
y_tcga = np.load('/home/leezoo/home/model_v2_dataset/test1(under1)/y(TCGA).npy')  # 예: (964,)

# TCGA DataFrame 생성
df_tcga = pd.DataFrame({
    'genes': list(X_gene_tcga),  # 리스트 형태로 저장
    'clinical': list(X_clinical_tcga),
    'response': y_tcga,
    'domain': DOMAIN_LABELS['TCGA']
})

# 다른 코호트 데이터 로드 및 더미 값 할당
cohorts = ['Liu', 'Hugo', 'VanAllen', 'Snyder']
df_list = [df_tcga]

for cohort in cohorts:
    X_gene = np.load(f'/home/leezoo/home/model_v2_dataset/test1(under1)/X_gene_data({cohort}).npy')  # 예: (N_cohort, 16856, 3)
    #y = np.load(f'/home/leezoo/home/model_v2_dataset/test1/y({cohort}).npy')  # 예: (N_cohort,)
    N = X_gene.shape[0]
    # 임상 데이터와 반응 레이블을 더미 값으로 할당
    X_clinical = np.zeros((N, 28), dtype=np.float32) # tcga clinical data 차원 28
    y = np.full((N,), -1, dtype=np.float32)  # -1은 반응 레이블이 없음을 의미
    df_cohort = pd.DataFrame({
        'genes': list(X_gene),
        'clinical': list(X_clinical),
        'response': y,
        'domain': DOMAIN_LABELS[cohort]
    })
    df_list.append(df_cohort)

# 모든 코호트 데이터 통합
df_all = pd.concat(df_list, ignore_index=True)
print(f"Combined dataset shape: {df_all.shape}")# (964 + N_liu + N_hugo + N_vanallen + N_snyder, ...)

# MASK MATRIX 생성 ########################################################################################################################
unique_pathways = set(pathway_dict.keys()) 
for child_list in pathway_dict.values():
    unique_pathways.update(child_list)  

pathway_index = {pathway: idx for idx, pathway in enumerate(unique_pathways)}

mask_matrices = [None] * 10
mask_matrices[0] = gene_to_pathway_matrix  # (gene 수 , pathway 수)

selected_pathways = list(pathway_dict.keys())[:2512]
pathway_index = {pathway: idx for idx, pathway in enumerate(selected_pathways)}

for i in range(1, 10):
    connected_parents = set()
    for parent_pathway, child_pathways in pathway_dict.items():
        for child_pathway in child_pathways:
            if child_pathway in pathway_index: 
                connected_parents.add(parent_pathway)
                break  
    
    all_related_pathways = list(connected_parents)
    full_pathway_index = {pathway: idx for idx, pathway in enumerate(all_related_pathways)}
    
    num_rows = len(pathway_index)  
    num_columns = len(full_pathway_index) 
    mask_matrices[i] = np.zeros((num_rows, num_columns), dtype=int)
    
    for parent_pathway, child_pathways in pathway_dict.items():
        if parent_pathway in full_pathway_index: 
            for child_pathway in child_pathways:
                if child_pathway in pathway_index:  
                    parent_index = full_pathway_index[parent_pathway]
                    child_index = pathway_index[child_pathway]
                    mask_matrices[i][child_index, parent_index] = 1 
    
    pathway_index = full_pathway_index


mask_matrices = [
    mask_matrices[0].values if isinstance(mask_matrices[0], pd.DataFrame) else mask_matrices[0]
] + mask_matrices[1:]


mask_matrices = [
    np.array(m) if not isinstance(m, np.ndarray) else m 
    for m in mask_matrices
]

print("Mask matrices shapes:")
for i, m in enumerate(mask_matrices):
    print(f"Mask {i} shape:", m.shape)

if isinstance(mask_matrices[1], pd.DataFrame):
    mask_matrices[0] = torch.tensor(mask_matrices[0].to_numpy(), dtype=torch.float32).to(device)
    mask_matrices[1] = torch.tensor(mask_matrices[1].to_numpy(), dtype=torch.float32).to(device) 
    mask_matrices[2] = torch.tensor(mask_matrices[2].to_numpy(), dtype=torch.float32).to(device) 
    mask_matrices[3] = torch.tensor(mask_matrices[3].to_numpy(), dtype=torch.float32).to(device)
    mask_matrices[4] = torch.tensor(mask_matrices[4].to_numpy(), dtype=torch.float32).to(device) 
  
class MultiDomainDataset(Dataset):
    def __init__(self, df):
        """
        df: 통합된 DataFrame, 컬럼 ['genes', 'clinical', 'response', 'domain']
        """
        self.genes = np.stack(df['genes'].values)  # Shape: (N, 16856, 3)
        self.clinicals = np.stack(df['clinical'].values)  # Shape: (N, 38)
        self.responses = df['response'].values  # Shape: (N,)
        self.domains = df['domain'].values  # Shape: (N,)
        
    def __len__(self):
        return len(self.genes)
    
    def __getitem__(self, idx):
        gene = torch.tensor(self.genes[idx], dtype=torch.float32)  # Shape: (16856, 3)
        clinical = torch.tensor(self.clinicals[idx], dtype=torch.float32)  # Shape: (38,)
        response = self.responses[idx]
        domain = self.domains[idx]
        
        # 반응 레이블을 텐서로 변환
        response = torch.tensor(response, dtype=torch.float32)
        # 도메인 레이블을 정수 텐서로 변환
        domain = torch.tensor(domain, dtype=torch.long)
        
        return gene, clinical, response, domain
# model v2 ########################################################################################################################
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


class BiOXNetEncoder(nn.Module):
    def __init__(self, mask_matrices, dropout_rate=0.3):
        super(BiOXNetEncoder, self).__init__()
        self.mask_matrices = [torch.FloatTensor(m).to(device) for m in mask_matrices]
        self.attention_weights = None
        
        # 첫 번째 레이어 파라미터
        self.W1 = nn.Parameter(torch.randn(6640, 3).to(device)) #랜덤하게 0~1사이 초기값
        self.b1 = nn.Parameter(torch.zeros(6640, 3).to(device)) #bias는 초기값을 0으로
        self.W1_a = nn.Parameter(torch.randn(6640, 3).to(device)) 
        self.b1_a = nn.Parameter(torch.zeros(6640, 3).to(device))
        
        # 나머지 레이어들의 파라미터
        self.layers = nn.ModuleList()
        dims = [ #추가 hidden layer 4개 (2512,1169,628,387) 
            (2512, 1169),
            (1169, 628),
            (628, 387),
        ]
        
        for in_dim, out_dim in dims:
            layer_params = {
                'W': nn.Parameter(torch.randn(in_dim, out_dim).to(device)),
                'b': nn.Parameter(torch.zeros(out_dim).to(device)),
                'W_a': nn.Parameter(torch.randn(in_dim, out_dim).to(device)),
                'b_a': nn.Parameter(torch.zeros(out_dim).to(device))
            }
            self.layers.append(nn.ParameterDict(layer_params))
        
        self.dropout = nn.Dropout(dropout_rate)
        
        all_dims = [6640, 2512, 1169, 628, 387]
        self.batch_norm_layers = nn.ModuleList([
            nn.BatchNorm1d(dim) for dim in all_dims
        ])
        
        

    def forward(self, x_gene):
        # gene_data 처리
        I = torch.ones(3, 1).to(device)
        xw = x_gene * self.W1
        xw = xw + self.b1
        O1_prime = torch.matmul(xw, I).squeeze(-1)
        
        xw_a = x_gene * self.W1_a
        xw_a = xw_a + self.b1_a
        pi1 = torch.sigmoid(torch.matmul(xw_a, I).squeeze(-1))
        
        self.attention_weights = pi1
        
        O1 = torch.tanh(O1_prime + pi1 * O1_prime)
        O1 = self.dropout(O1)
        O1 = O1.view(O1.size(0), -1)
        O1 = self.batch_norm_layers[0](O1)
        
        O = torch.matmul(O1, self.mask_matrices[0])
        O = self.batch_norm_layers[1](O)
        
        for i, layer_params in enumerate(self.layers):
            Mi = self.mask_matrices[i+1]
            MW = Mi * layer_params['W']
            Oi_prime = torch.matmul(O, MW) + layer_params['b']
            pi_i = torch.sigmoid(torch.matmul(O, layer_params['W_a']) + layer_params['b_a'])
            O = torch.tanh(Oi_prime + pi_i * Oi_prime)
            O = self.dropout(O)
            O = self.batch_norm_layers[i+2](O)
        
        
        return O
    
    
class DomainClassifier(nn.Module):
    def __init__(self, input_dim=387, num_domains=5):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains)  # Softmax는 Loss에서 처리
        )
    def forward(self, x):
        return self.classifier(x)  # (batch_size, num_domains)  
    
class MultiTaskModel(nn.Module):
    def __init__(self, mask_matrices, dropout_rate=0.3, alpha=1.0):
        super(MultiTaskModel, self).__init__()
        
        # 5.1 Shared Encoder
        self.encoder = BiOXNetEncoder(mask_matrices, dropout_rate=dropout_rate)
        
        # 5.2 임베딩(Clinical Data)
        self.cancer_type_embedding = nn.Embedding(19, 10)  # 예: 암종 19
        self.treatment_type_embedding = nn.Embedding(6, 3) # 예: 치료 6
        
        # 5.3 Response Classifier
        #  (shared_feature=387) + (cancer=10 + treatment=3 + gender=2 + age=1) => 403
        self.response_classifier = nn.Sequential(
            nn.Linear(387 + 10 + 3 + 2 + 1, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 5.4 Domain Classifier (5-class)
        self.domain_classifier = DomainClassifier(input_dim=387, num_domains=5)
        
        # GRL 세기
        self.alpha = alpha

    def forward(self, x_gene, x_clinical, enable_grl=True):
        """
        x_gene: (batch_size, 10518)
        x_clinical: (batch_size, 28)  # 예시 (각자 실제 형태 맞게 수정)
        
        enable_grl: 학습 시 True => GRL 사용, 
                    추론 시 False => GRL 미사용
        """
        # 1) Shared Encoder
        shared_feature = self.encoder(x_gene)  # (batch_size, 387)
        
        # 2) Response Classifier
        # 암종(19개 중 idx), 치료(6개 중 idx), gender(2 one-hot), age(단일 스칼라) 추출
        cancer_type = x_clinical[:, :19].argmax(dim=1)
        treatment_type = x_clinical[:, 21:27].argmax(dim=1)
        gender = x_clinical[:, 19:21]  # 2차원
        age = x_clinical[:, 27:28]     # 1차원
        
        cancer_embedding = self.cancer_type_embedding(cancer_type)
        treatment_embedding = self.treatment_type_embedding(treatment_type)
        
        combined = torch.cat(
            (shared_feature, cancer_embedding, treatment_embedding, gender, age),
            dim=1
        )
        response_pred = self.response_classifier(combined)  # (batch_size, 1)
        
        # 3) Domain Classifier (Adversarial)
        if enable_grl:
            reverse_feature = grad_reverse(shared_feature, self.alpha)
        else:
            reverse_feature = shared_feature
        
        domain_logits = self.domain_classifier(reverse_feature)  # (batch_size, 5)
        
        return response_pred, domain_logits



# 손실 함수 설정
criterion_response = nn.BCELoss()
criterion_domain = nn.CrossEntropyLoss()

  
# 데이터 분할: 도메인별로 균등하게 분할
train_df, temp_df = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['domain'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['domain'])

# 분할된 데이터셋 저장
train_df.to_csv('/home/leezoo/home/model_v2_dataset/test1(under1)/train_data.csv', index=False)
val_df.to_csv('/home/leezoo/home/model_v2_dataset/test1(under1)/val_data.csv', index=False)
test_df.to_csv('/home/leezoo/home/model_v2_dataset/test1(under1)/test_data.csv', index=False)

# Dataset 생성
train_dataset = MultiDomainDataset(train_df)
val_dataset = MultiDomainDataset(val_df)
test_dataset = MultiDomainDataset(test_df)

# 데이터셋 크기 확인
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")





# 결과 저장 경로 설정
RESULT_DIR = '/home/leezoo/home/model_v2_dataset/test1(under1)/result'
os.makedirs(RESULT_DIR, exist_ok=True)

# 하이퍼파라미터 설정 ##################################################################################################################################################################################################################
batch_sizes = [5,30,500, 816] #훈련에서의 배치 사이즈
learning_rates = [0.001, 0.0001, 0.00005]
EPOCHS = 1500
lambda_domain = 1.0  # 도메인 손실의 가중치 (필요에 따라 조정)


#####################################################################################################################################################################


# 최종 validation 결과 저장을 위한 딕셔너리
final_val_results = {}
# 모델 초기화 및 초기 가중치 저장
initial_model = MultiTaskModel(mask_matrices, dropout_rate=0.3, alpha=1.0).to(device)
initial_state_dict = initial_model.state_dict()

# 학습 전 가중치 저장
torch.save(initial_model.encoder.state_dict(), '/home/leezoo/home/model_v2_dataset/test1(under1)/result/pretrained_encoder.pth')

# 각 하이퍼파라미터 조합에 대해 학습
for batch_size in batch_sizes:
    for lr in learning_rates:
        model_name = f'model_bs{batch_size}_lr{lr}'
        print(f"\n훈련 시작: batch_size={batch_size}, learning_rate={lr}")
        
        # 모델 복제 및 초기 가중치 로드
        model = MultiTaskModel(mask_matrices, dropout_rate=0.3, alpha=1.0).to(device)
        model.load_state_dict(initial_state_dict)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        
        # DataLoader 생성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4) #검증 단계에선 배치 작게
        # (테스트 로더 추가)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

        # 학습 및 검증
        train_response_losses = []
        val_response_losses = []
        train_domain_losses = []
        
        # === Training Loop ===
    
        for epoch in tqdm(range(EPOCHS), desc=f"Training {model_name}", leave=False):
            model.train()
            epoch_response_loss = 0.0
            epoch_domain_loss = 0.0
            for batch in train_loader:
                gene, clinical, response, domain = batch
                gene = gene.to(device)
                clinical = clinical.to(device)
                response = response.to(device)
                domain = domain.to(device)

                optimizer.zero_grad()
                response_pred, domain_pred = model(gene, clinical, enable_grl=True)

                # TCGA 코호트에 대해서만 반응 예측 손실 계산
                tcga_mask = (domain == DOMAIN_LABELS['TCGA'])
                if tcga_mask.sum() > 0:
                    response_pred_tcga = response_pred[tcga_mask]
                    response_tcga = response[tcga_mask].unsqueeze(1)
                    loss_response = criterion_response(response_pred_tcga, response_tcga)
                else:
                    loss_response = torch.tensor(0.0, device=device)

                loss_domain = criterion_domain(domain_pred, domain)
                loss = loss_response + lambda_domain * loss_domain
                loss.backward()
                optimizer.step()

                epoch_response_loss += loss_response.item()
                epoch_domain_loss += loss_domain.item()

            avg_epoch_response_loss = epoch_response_loss / len(train_loader)
            avg_epoch_domain_loss = epoch_domain_loss / len(train_loader)
            train_response_losses.append(avg_epoch_response_loss)
            train_domain_losses.append(avg_epoch_domain_loss)

            # === Validation Loop ===
            model.eval()
            val_response_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    gene, clinical, response, domain = batch
                    gene = gene.to(device)
                    clinical = clinical.to(device)
                    response = response.to(device)
                    domain = domain.to(device)

                    response_pred, domain_pred = model(gene, clinical, enable_grl=False)
                    
                    # TCGA 코호트에 대해서만 반응 예측 손실 계산
                    tcga_mask = (domain == DOMAIN_LABELS['TCGA'])
                    if tcga_mask.sum() > 0:
                        response_pred_tcga = response_pred[tcga_mask]
                        response_tcga = response[tcga_mask].unsqueeze(1)
                        loss_response = criterion_response(response_pred_tcga, response_tcga)
                        val_response_loss += loss_response.item()

            avg_val_response_loss = val_response_loss / len(val_loader)
            val_response_losses.append(avg_val_response_loss)

        # 1) Validation 성능 측정
        model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch in val_loader:
                gene, clinical, response, domain = batch
                gene = gene.to(device)
                clinical = clinical.to(device)
                response = response.to(device)
                domain = domain.to(device)

                response_pred, domain_pred = model(gene, clinical, enable_grl=False)

                tcga_mask = (domain == DOMAIN_LABELS['TCGA'])
                if tcga_mask.sum() > 0:
                    response_pred_tcga = response_pred[tcga_mask]
                    response_tcga = response[tcga_mask].unsqueeze(1)

                    val_preds.extend((response_pred_tcga > 0.5).cpu().numpy())
                    val_true.extend(response_tcga.cpu().numpy())

        # 모델 이름 선언
        model_name = f'model_bs{batch_size}_lr{lr}'

        # 혼동 행렬 생성 및 저장
        val_cm = confusion_matrix(val_true, val_preds)
        val_cm_normalized = val_cm.astype('float') / val_cm.sum(axis=1)[:, np.newaxis]  # 행별로 정규화

        plt.figure(figsize=(10, 7))
        sns.heatmap(val_cm_normalized, annot=True, fmt='.2f', cmap='Blues')  # fmt를 '.2f'로 변경하여 소수점 표시
        plt.title(f'Final Confusion Matrix (bs={batch_size}, lr={lr})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(RESULT_DIR, f'{model_name}_confusion_matrix.png'))
        plt.close()

        # 성능 지표 계산
        val_accuracy = accuracy_score(val_true, val_preds)
        val_precision = precision_score(val_true, val_preds, average='binary', zero_division=0)
        val_recall = recall_score(val_true, val_preds, average='binary', zero_division=0)
        val_f1 = f1_score(val_true, val_preds, average='binary', zero_division=0)

        # 2) Test 성능 측정 (추가)
        test_preds = []
        test_true = []
        with torch.no_grad():
            for batch in test_loader:
                gene, clinical, response, domain = batch
                gene = gene.to(device)
                clinical = clinical.to(device)
                response = response.to(device)
                domain = domain.to(device)

                response_pred, domain_pred = model(gene, clinical, enable_grl=False)
                
                tcga_mask = (domain == DOMAIN_LABELS['TCGA'])
                if tcga_mask.sum() > 0:
                    response_pred_tcga = response_pred[tcga_mask]
                    response_tcga = response[tcga_mask].unsqueeze(1)
                    test_preds.extend((response_pred_tcga > 0.5).cpu().numpy())
                    test_true.extend(response_tcga.cpu().numpy())

        test_cm = confusion_matrix(test_true, test_preds)
        test_cm_normalized = test_cm.astype('float') / test_cm.sum(axis=1, keepdims=True)
        plt.figure(figsize=(6, 5))
        sns.heatmap(test_cm_normalized, annot=True, fmt='.2f', cmap='Greens')
        plt.title(f'Test Confusion Matrix (bs={batch_size}, lr={lr})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(RESULT_DIR, f'{model_name}_test_confusion_matrix.png'))
        plt.close()

        test_accuracy = accuracy_score(test_true, test_preds)
        test_precision = precision_score(test_true, test_preds, average='binary', zero_division=0)
        test_recall = recall_score(test_true, test_preds, average='binary', zero_division=0)
        test_f1 = f1_score(test_true, test_preds, average='binary', zero_division=0)



        # 모델 가중치 저장
        torch.save(model.state_dict(), os.path.join(RESULT_DIR, f'{model_name}.pth'))

        # Shared Encoder 가중치 저장
        shared_encoder_weights = {k: v.cpu().numpy() for k, v in model.state_dict().items() if 'encoder' in k}
        np.savez(os.path.join(RESULT_DIR, f'{model_name}_trained_encoder.npz'), **shared_encoder_weights)
       
        
        # 손실 플롯 저장
        plt.figure(figsize=(10, 5))
        plt.plot(train_response_losses, label='Train Response Loss')
        plt.plot(val_response_losses, label='Validation Response Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Response Loss Plot (Batch Size: {batch_size}, LR: {lr})')
        plt.legend()
        plt.savefig(os.path.join(RESULT_DIR, f'response_loss_plot_bs{batch_size}_lr{lr}.png'))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(train_domain_losses, label='Train Domain Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Domain Loss Plot (Batch Size: {batch_size}, LR: {lr})')
        plt.legend()
        plt.savefig(os.path.join(RESULT_DIR, f'domain_loss_plot_bs{batch_size}_lr{lr}.png'))
        plt.close()

        # 결과 저장
        final_val_results[model_name] = {
            'batch_size': batch_size,
            'learning_rate': lr,

            # Validation 성능
            'final_val_accuracy': val_accuracy,
            'final_val_precision': val_precision,
            'final_val_recall': val_recall,
            'final_val_f1': val_f1,

            # Test 성능 (추가)
            'final_test_accuracy': test_accuracy,
            'final_test_precision': test_precision,
            'final_test_recall': test_recall,
            'final_test_f1': test_f1
        }

# 최종 validation 결과 저장
with open(os.path.join(RESULT_DIR, 'final_validation_results.json'), 'w') as f:
    json.dump(final_val_results, f, indent=4)

# 결과를 표 형태로 출력
print("\n=== 모든 모델의 최종 Validation 결과 ===")
print("Model Name | Batch Size | Learning Rate | Accuracy | Precision | Recall | F1")
print("-" * 80)
for model_name, results in final_val_results.items():
    print(f"{model_name} | {results['batch_size']} | {results['learning_rate']} | "
          f"{results['final_val_accuracy']:.4f} | {results['final_val_precision']:.4f} | "
          f"{results['final_val_recall']:.4f} | {results['final_val_f1']:.4f}")





print(model)