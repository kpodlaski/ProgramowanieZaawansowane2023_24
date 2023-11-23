from sklearn.datasets import load_digits
from sklearn.model_selection._split import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm

digits = load_digits()
print(digits.data.shape)

plt.gray()
plt.imshow(digits.images[32])
print(digits.images[32].shape)
print(digits.target[32])
plt.show()
X_train, X_test, y_train, y_test = train_test_split(digits.images, digits.target,
                test_size=0.2)

plt.imshow(X_train[32])
print(y_train[32])
plt.show()
clf = svm.SVC(kernel='linear')
X_train = X_train.reshape(1437,-1)
print(X_train.shape)
clf.fit(X_train, y_train)
# Testowanie na zbiorze uczącym:
y_pred = clf.predict(X_train)
poprawne = 0
for i in range(len(y_pred)):
    if y_pred[i]==y_train[i]:
        poprawne+=1

print("Acc:",poprawne/len(y_pred))
# Testowanie na zbiorze testowym:
y_pred = clf.predict(X_test.reshape(360,-1))
poprawne = 0
for i in range(len(y_pred)):
    if y_pred[i]==y_test[i]:
        poprawne+=1
    else:
        plt.imshow(X_test[i])
        plt.title("Pred:"+str(y_pred[i]) + " act:"+str(y_test[i]))
        plt.show()
print("Acc:",poprawne/len(y_pred))
print("ilość błędów:",360-poprawne)