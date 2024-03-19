#include <iostream>
#include <vector>
#include <cmath>
#include <map>

using namespace std;

// Constants
const double dt = 1.0; // Time step
const double sigma_a = 0.1; // Process noise standard deviation for acceleration
const double sigma_measurement = 0.5; // Measurement noise standard deviation

// Kalman Filter Class
class KalmanFilter {
private:
    // State vector: [x, y, z, vx, vy, vz]
    vector<double> x;
    
    // State transition matrix
    vector<vector<double>> F;
    
    // Measurement matrix
    vector<vector<double>> H;
    
    // Process covariance matrix
    vector<vector<double>> Q;
    
    // Measurement covariance matrix
    vector<vector<double>> R;
    
public:
    // Constructor
    KalmanFilter() {
        // Initialize state vector
        x = vector<double>(6, 0.0);
        
        // Initialize state transition matrix
        F = {{1, 0, 0, dt, 0, 0},
             {0, 1, 0, 0, dt, 0},
             {0, 0, 1, 0, 0, dt},
             {0, 0, 0, 1, 0, 0},
             {0, 0, 0, 0, 1, 0},
             {0, 0, 0, 0, 0, 1}};
        
        // Initialize measurement matrix
        H = {{1, 0, 0, 0, 0, 0},
             {0, 1, 0, 0, 0, 0},
             {0, 0, 1, 0, 0, 0}};
        
        // Initialize process covariance matrix
        Q = {{pow(dt, 4)/4 * pow(sigma_a, 2), 0, 0, pow(dt, 3)/2 * pow(sigma_a, 2), 0, 0},
             {0, pow(dt, 4)/4 * pow(sigma_a, 2), 0, 0, pow(dt, 3)/2 * pow(sigma_a, 2), 0},
             {0, 0, pow(dt, 4)/4 * pow(sigma_a, 2), 0, 0, pow(dt, 3)/2 * pow(sigma_a, 2)},
             {pow(dt, 3)/2 * pow(sigma_a, 2), 0, 0, pow(dt, 2) * pow(sigma_a, 2), 0, 0},
             {0, pow(dt, 3)/2 * pow(sigma_a, 2), 0, 0, pow(dt, 2) * pow(sigma_a, 2), 0},
             {0, 0, pow(dt, 3)/2 * pow(sigma_a, 2), 0, 0, pow(dt, 2) * pow(sigma_a, 2)}};
        
        // Initialize measurement covariance matrix
        R = {{pow(sigma_measurement, 2), 0, 0},
             {0, pow(sigma_measurement, 2), 0},
             {0, 0, pow(sigma_measurement, 2)}};
    }
    
    // Predict step of Kalman Filter
    void predict() {
        // Predict state
        x = matrix_multiply(F, x);
        
        // Predict error covariance
        vector<vector<double>> Ft = transpose(F);
        Q = matrix_add(Q, R);
        Q = matrix_multiply(F, matrix_multiply(Q, Ft));
    }
    
    // Update step of Kalman Filter
    void update(vector<double> z) {
        // Calculate Kalman gain
        vector<vector<double>> Ht = transpose(H);
        vector<vector<double>> S = matrix_add(matrix_multiply(H, matrix_multiply(Q, Ht)), R);
        vector<vector<double>> K = matrix_multiply(matrix_multiply(Q, Ht), matrix_inverse(S));
        
        // Update state estimate
        vector<double> y = matrix_subtract(z, matrix_multiply(H, x));
        x = matrix_add(x, matrix_multiply(K, y));
        
        // Update error covariance
        vector<vector<double>> I = identity_matrix(6);
        Q = matrix_multiply(matrix_subtract(I, matrix_multiply(K, H)), Q);
    }
    
    // Get predicted state
    vector<double> getState() {
        return x;
    }
    
    // Utility functions
    vector<vector<double>> matrix_multiply(vector<vector<double>>& A, vector<vector<double>>& B) {
        int m = A.size();
        int n = A[0].size();
        int p = B[0].size();
        
        vector<vector<double>> result(m, vector<double>(p, 0.0));
        
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                for (int k = 0; k < n; ++k) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        
        return result;
    }
    
    vector<vector<double>> matrix_add(vector<vector<double>>& A, vector<vector<double>>& B) {
        int m = A.size();
        int n = A[0].size();
        
        vector<vector<double>> result(m, vector<double>(n, 0.0));
        
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                result[i][j] = A[i][j] + B[i][j];
            }
        }
        
        return result;
    }
    
    vector<vector<double>> matrix_subtract(vector<vector<double>>& A, vector<vector<double>>& B) {
        int m = A.size();
        int n = A[0].size();
        
        vector<vector<double>> result(m, vector<double>(n, 0.0));
        
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                result[i][j] = A[i][j] - B[i][j];
            }
        }
        
        return result;
    }
    
    vector<vector<double>> transpose(vector<vector<double>>& A) {
        int m = A.size();
        int n = A[0].size();
        
        vector<vector<double>> result(n, vector<double>(m, 0.0));
        
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                result[j][i] = A[i][j];
            }
        }
        
        return result;
    }
    
    vector<vector<double>> matrix_inverse(vector<vector<double>>& A) {
        // Assuming A is 3x3 matrix
        double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                     A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                     A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
        
        vector<vector<double>> result(3, vector<double>(3, 0.0));
        result[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) / det;
        result[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) / det;
        result[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) / det;
        result[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) / det;
        result[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) / det;
        result[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) / det;
        result[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) / det;
        result[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) / det;
        result[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / det;
        
        return result;
    }
    
    vector<vector<double>> identity_matrix(int n) {
        vector<vector<double>> result(n, vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) {
            result[i][i] = 1.0;
        }
        return result;
    }
};

// Main function
int main() {
    // Sample input data
    vector<int> TN = {1, 2, 3}; // Target IDs
    vector<double> MR = {10.0, 12.0, 15.0}; // Measurement range
    vector<double> MA = {30.0, 45.0, 60.0}; // Measurement azimuth
    vector<double> ME = {5.0, 10.0, 15.0}; // Measurement elevation
    vector<double> MT = {0.0, 1.0, 2.0}; // Measurement time
    
    // Map to store predicted states for each target ID
    map<int, vector<double>> predictedStates;
    
    // Kalman filter object
    KalmanFilter kf;
    
    // Perform prediction and update for each measurement
    for (int i = 0; i < TN.size(); ++i) {
        // Predict
        kf.predict();
        
        // Update with measurement
        vector<double> z = {MR[i], MA[i], ME[i]};
        kf.update(z);
        
        // Get predicted state
        vector<double> predictedState = kf.getState();
        
        // Store predicted state for this target ID
        predictedStates[TN[i]] = predictedState;
    }
    
    // Print predicted states
    cout << "Predicted States:" << endl;
    for (const auto& entry : predictedStates) {
        int targetID = entry.first;
        vector<double> state = entry.second;
        cout << "Target ID: " << targetID << ", Predicted State: [x=" << state[0] << ", y=" << state[1] << ", z=" << state[2]
             << ", vx=" << state[3] << ", vy=" << state[4] << ", vz=" << state[5] << "]" << endl;
    }
    
    return 0;
}
