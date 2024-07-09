import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
print(matplotlib.__version__)
import matplotlib.pyplot as plt

def load_and_label_data(folder_path):
    file_paths = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith('.csv')]
    labels = []
    features_list = []

    for file in file_paths:
        data = np.loadtxt(join(folder_path, file), delimiter=',')
        # Label assignment based on 'x' column
        label = np.max(data[:, 0]) > 10.0
        labels.append(float(label))

        # Filter data where x < 5 and dimensions are not all zeros
        valid_data = data[(data[:, 0] < 5) & (data[:, 0] > -1.9) & (data[:, 4] != 0) & (data[:, 5] != 0) & (data[:, 6] != 0)]
        if valid_data.size > 0:
            mean_v = np.mean(valid_data[:, 2])
            mean_dim_x = np.mean(valid_data[:, 4])
            mean_dim_y = np.mean(valid_data[:, 5])
            mean_dim_z = np.mean(valid_data[:, 6])
            mean_eng_pw = np.mean(valid_data[:,-2]*valid_data[:,-1])
            features_list.append([mean_v, mean_dim_x, mean_dim_y, mean_dim_z,mean_eng_pw])
        else:
            # Append a placeholder if no valid data is found
            features_list.append([0, 0, 0, 0])  # Default values when conditions are not met

    return np.array(features_list), np.array(labels)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Define the layers
        self.layer1 = nn.Linear(5, 512)  # Input dimension is 5
        self.layer2 = nn.Linear(512, 128)
        self.layer3 = nn.Linear(128, 32)
        self.layer4 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        # Pass through the network layers
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.output(x)
        x = torch.sigmoid(x)  # Apply sigmoid activation to output
        return x



def train_model(features, labels):
    # Convert numpy arrays to torch tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # Model instantiation
    model = NeuralNetwork()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        outputs = model(features_tensor)
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    # Save the trained model
    torch.save(model.state_dict(), 'tracked_veh_cond_avoid.pth')

def load_model(model_path):
    # Initialize the model
    model = NeuralNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def make_inference(model, new_features):
    # Convert new features to torch tensor
    new_features_tensor = torch.tensor(new_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():  # No need to track gradients for inference
        output = model(new_features_tensor)
        predicted_prob = torch.sigmoid(output).item()
    return predicted_prob

# Load data
folder_path = 'training_data/M113trainingData'
features, labels = load_and_label_data(folder_path)
scaler = StandardScaler()
features = scaler.fit_transform(features)
np.savetxt('features.csv', features, delimiter=',')
np.savetxt('labels.csv', labels, delimiter=',')
# print(features.shape)
# print(labels.shape)

############## Train the model using NN ################
train_model(features, labels)

# Load the trained model
model_path = 'tracked_veh_cond_avoid.pth'
model = load_model(model_path)
predicted_label = []
for i in range(len(features)):
    prediction = make_inference(model, features[i])
    # if prediction > 0.6:
    #     predicted_label.append(1)
    # else:
    #     predicted_label.append(0)
    predicted_label.append(prediction)
# use clustering to separate the prediction 

predicted_label = np.array(predicted_label).reshape(-1, 1)
print(predicted_label)
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(predicted_label)
centroids = kmeans.cluster_centers_
print(centroids)
boundary = (centroids[0] + centroids[1]) / 2  
print(boundary)

# count the number of correct predictions
correct = 0
for i in range(len(predicted_label)):
    if predicted_label[i] > boundary:
        #predicted_label[i] = 1
        if labels[i] == 1:
            correct += 1
    else:
        #predicted_label[i] = 0
        if labels[i] == 0:
            correct += 1

print(f'Accuracy: {correct/len(predicted_label)}')

# print('length of predicted label:', len(features))
# print('length of predicted label:', len(labels))
# print('length of predicted label:', len(predicted_label))

plt.figure()
plt.plot([0, len(features)], [boundary, boundary], 'r--')
plt.scatter(range(len(features)), labels, label='True Label')
plt.scatter(range(len(features)), predicted_label, label='Predicted Label')
plt.legend()
plt.show()