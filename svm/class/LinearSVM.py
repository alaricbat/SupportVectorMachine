import numpy as np

class LinearSVM:

    def __init__(self,
                 n_iters = 3000, 
                 learning_rate = 0.0001,
                 C = 1.0,
                 l1_lambda= 0.01,):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.C = C
        self.l1_lambda = l1_lambda
        self.w = None
        self.b = None

    def fit(self, X, y):

        # 1. Convert targets {0, 1} to {-1, 1} for standard SVM mathematical formulation
        y_svm = np.where(y <=0, -1, 1)
        n_features = X.shape[1]

        # 2. Initialize weights and bias to zeros
        self.w = np.zeros(n_features)
        self.b = 0.0

        n_class_0 = np.sum(y_svm == -1)
        n_class_1 = np.sum(y_svm == 1)
        weight_1 = n_class_0 / n_class_1 if n_class_1 > 0 else 1.0

        for _ in range(self.n_iters):

            for i, xi in enumerate(X):

                grad_w = 0.0 
                grad_b = 0.0

                # Calculate functional margin: y_i * (w · x_i + b)
                functional_margin = y_svm[i] * (np.dot(xi, self.w) + self.b)

                # Gradient of (Original Margin Objective 1/2||w||^2 + L1 Penalty)
                grad_objective = self.w + self.l1_lambda * np.sign(self.w)

                if functional_margin >= 1:
                    # Sample satisfies margin condition -> No Hinge Loss gradient
                    grad_w = grad_objective
                    grad_b = 0.0
                else:
                    # Sample violates margin condition -> Include Hinge Loss gradient
                    c_weight = self.C * (weight_1 if y_svm[i] == 1 else 1.0)
                    grad_w = grad_objective - c_weight * y_svm[i] * xi
                    grad_b = -c_weight * y_svm[i]

                self.w -= self.learning_rate * grad_w
                self.b -= self.learning_rate * grad_b
            


    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.where(approx >= 0, 1, 0)
