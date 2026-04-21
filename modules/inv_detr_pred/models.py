import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseMatcher(nn.Module):
    def __init__(self, num_classes):
        super(InverseMatcher, self).__init__()
        self.num_classes = num_classes

    def forward(self, classes):
        matched_classes = torch.argmax(classes, dim=2)
        class_logits = F.one_hot(matched_classes, num_classes=self.num_classes + 1).float()
        return class_logits


class InverseClassifierRegressor(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=512, embedding_dim=256):
        super(InverseClassifierRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, embedding_dim)

    def forward(self, class_logits, bboxes):
        x = torch.cat([class_logits, bboxes], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        embeddings = self.fc3(x)
        return embeddings


class InverseDetector(nn.Module):
    def __init__(self, num_classes=91, bbox_dim=4, hidden_dim=512, embedding_dim=256):
        super(InverseDetector, self).__init__()
        self.inverse_matcher = InverseMatcher(num_classes)
        input_dim = num_classes + 1 + bbox_dim
        self.inverse_classifier_regressor = InverseClassifierRegressor(input_dim, hidden_dim, embedding_dim)

    def forward(self, pred_classes, pred_bboxes):
        class_logits = self.inverse_matcher(pred_classes)
        embeddings = self.inverse_classifier_regressor(class_logits, pred_bboxes)

        return embeddings
