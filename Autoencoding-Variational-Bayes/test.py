import torch
import torch.nn
from torch.autograd import Variable

from utils import loss_function

# Inference 모델 정의
def test(model, testset, cuda = True):
    '''
    train과 동일한 방법으로 loss_function을 계산, 즉,
     (1) 학습된 모델로 input image를 reconstruct하고 원본 이미지와 비교
     (2) 추정된 q(z|x)와 N(0, 1) 간 거리를 비교
    하여 test_loss 계산
    '''
    testset = testset
    
    device = torch.device("cuda" if cuda else "cpu")
    
    test_loss = 0.0
    
    model.eval()
    with torch.no_grad(): # gradients를 freezing하여 inference만 수행
        
        for batch_idx, (images, _) in enumerate(testset):
            
            images = Variable(images).to(device) if cuda else Variable(images)
            reconstructed, mu, logvar = model(images)
            
            test_loss += loss_function(images, reconstructed, mu, logvar).item()
            
        return test_loss / len(testset.dataset)