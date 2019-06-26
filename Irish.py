from sklearn.datasets import load_wine
data = load_wine()
print(data.target[[10, 80, 140]])
print(list(data.target_names))





