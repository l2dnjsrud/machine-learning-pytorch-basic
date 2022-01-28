import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

#모델 초기화
W = torch.zeros((4,3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    #cost 계산
    z = x_train.matmul(W) + b ## or .mm or @
    cost = F.cross_entropy(z, y_train)

    #cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    #100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch{:4d}/ {} cost{:.6f}'.format(
            epoch, nb_epochs, cost.item()
        )) 