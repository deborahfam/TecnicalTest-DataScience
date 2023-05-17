from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def SVM_model(df):     
       # Initialization of the SVM classifier
        clf = SVC(kernel='linear', C=1.0)

        # Generate new dataset or modify training data if necessary.
        X_train, X_test, y_train, y_test = train_test_split(df[['Number']], df['Label'], test_size=0.2, random_state=42)
            
        # Train the model at each epoch
        clf.fit(X_train, y_train)
            
        # Final evaluation of the model on the test set
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
        #print("Final Accuracy:", accuracy)
