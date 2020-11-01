import os
import torch
import torch.nn
from torch.autograd import Variable

from utils import loss_softmax, loss_svm

# Inference 모델 정의
def test(model, testset, test_type = 'softmax', cuda = True, model_path = './model'):
    '''
    '''
    model = model
    testset = testset

    device = torch.device("cuda" if cuda else "cpu")
    
    pred_correct = 0
    
    model.eval()
    with torch.no_grad(): # gradients를 freezing하여 inference만 수행
        
        for batch_idx, (images, labels) in enumerate(testset):
            
            images = Variable(images).to(device) if cuda else Variable(images)
            labels = Variable(labels).to(device) if cuda else Variable(labels)

            pred, _ = model(images)
            
            # 정확도 계산
            pred_label = torch.argmax(pred, axis = 1)
            
            pred_tf = torch.sum(torch.eq(pred_label, labels))
            pred_correct += pred_tf

        return pred_correct / len(testset.dataset)