from ch03.multi_linear_regression import Hypothesis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#For reproducibilith
torch.manual_seed(1)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forwawrd(self, x):
        return self.sigmoid(self.linear(x))

model = BinaryClassifier()

#optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 100
for epoch in range(nb_epochs + 1):

    #H(x) 계산
    Hypothesis = model(x_train)

    #cost 계산
    cost = F.binary_cross_entropy(Hypothesis, y_train)

    #cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    #20번마다 로그 출력
    if epoch % 10 == 0:
        prediction = Hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {:4d}/{} cost{: .6f} Accuracy {:2.2f}%'.format)(
            epoch, nb_epochs, cost.item(), accuracy * 100,
        )