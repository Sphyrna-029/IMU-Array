import numpy as np
from scipy.linalg import block_diag

class IMUKalmanFilter:
    def __init__(self, num_sensors=9, dt=0.01, process_noise=0.01, measurement_noise=0.1):
        """
        Initialize the Kalman filter for IMU sensor fusion.
        
        Parameters:
        - num_sensors: Number of IMU sensors (default: 9)
        - dt: Time step between measurements in seconds (default: 0.01)
        - process_noise: Process noise coefficient (default: 0.01)
        - measurement_noise: Measurement noise coefficient (default: 0.1)
        """
        self.num_sensors = num_sensors
        self.dt = dt
        
        # State vector: [x, y, z, vx, vy, vz, ax, ay, az]
        self.state_dim = 9
        self.measurement_dim = 3 * num_sensors  # 3 axes (accel) per IMU
        
        # Initialize state
        self.x = np.zeros((self.state_dim, 1))
        
        # State transition matrix (F)
        self.F = np.eye(self.state_dim)

        # Position from velocity: x += vx*dt
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        
        # Velocity from acceleration: vx += ax*dt
        self.F[3, 6] = dt
        self.F[4, 7] = dt
        self.F[5, 8] = dt
        
        # Process noise covariance matrix (Q)
        # Using discretized continuous white noise model
        q_pos = process_noise * dt**3 / 3  # Position variance
        q_vel = process_noise * dt**2 / 2  # Velocity variance
        q_acc = process_noise * dt         # Acceleration variance
        
        self.Q = np.zeros((self.state_dim, self.state_dim))
        # Position variance terms
        self.Q[0, 0] = q_pos
        self.Q[1, 1] = q_pos
        self.Q[2, 2] = q_pos
        # Velocity variance terms
        self.Q[3, 3] = q_vel
        self.Q[4, 4] = q_vel
        self.Q[5, 5] = q_vel
        # Acceleration variance terms
        self.Q[6, 6] = q_acc
        self.Q[7, 7] = q_acc
        self.Q[8, 8] = q_acc
        
        # Measurement matrix (H)
        # We directly measure accelerations from IMUs
        self.H = np.zeros((self.measurement_dim, self.state_dim))
        for i in range(num_sensors):
            # Each IMU gives us acceleration readings for x, y, z
            self.H[i*3:(i+1)*3, 6:9] = np.eye(3)
        
        # Measurement noise covariance (R)
        self.R = np.eye(self.measurement_dim) * measurement_noise
        
        # Initial error covariance matrix (P)
        self.P = np.eye(self.state_dim)
        
        # Identity matrix (I)
        self.I = np.eye(self.state_dim)
    
    def predict(self):
        """
        Prediction step of the Kalman filter.
        """
        # Project state ahead
        self.x = np.dot(self.F, self.x)
        
        # Project error covariance ahead
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return self.x
    
    def update(self, measurements):
        """
        Update step of the Kalman filter.
        
        Parameters:
        - measurements: Array of shape (num_sensors*3, 1) containing accelerometer
                       readings from all IMU sensors
        """
        # Reshape measurements if needed
        if len(measurements.shape) == 1:
            measurements = measurements.reshape(-1, 1)
        
        # Compute Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update estimate with measurement
        y = measurements - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        
        # Update error covariance
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)
        
        return self.x
    
    def process_imu_data(self, imu_data):
        """
        Process IMU data and return position estimate.
        
        Parameters:
        - imu_data: Array of shape (num_sensors, 3) containing accelerometer
                   readings from all IMU sensors
        
        Returns:
        - Estimated position [x, y, z]
        """
        # Flatten IMU data for measurement update
        measurements = imu_data.flatten().reshape(-1, 1)
        
        # Make prediction
        self.predict()
        
        # Update with measurements
        self.update(measurements)
        
        # Return position components (x, y, z)
        return self.x[0:3, 0]


def track_position_with_imus(imu_data_stream, dt=0.01):
    """
    Track position using data from 9 IMU sensors.
    
    Parameters:
    - imu_data_stream: List/generator of IMU readings, where each reading is a numpy array
                      of shape (9, 3) representing data from 9 IMU sensors (x, y, z accelerations)
    - dt: Time step between measurements in seconds
    
    Returns:
    - List of position estimates [x, y, z] for each time step
    """
    # Initialize Kalman filter
    kf = IMUKalmanFilter(num_sensors=9, dt=dt)
    
    # Process IMU data stream
    positions = []
    for imu_data in imu_data_stream:
        # Validate input data shape
        if imu_data.shape != (9, 3):
            raise ValueError(f"Expected IMU data shape (9, 3), got {imu_data.shape}")
        
        # Process IMU data
        position = kf.process_imu_data(imu_data)
        positions.append(position)
    
    return np.array(positions)


# Example usage
if __name__ == "__main__":
    # Simulate 100 time steps of IMU data
    np.random.seed(42)
    num_steps = 100
    
    # Simulate a simple trajectory (circular motion with noise)
    t = np.linspace(0, 10, num_steps)
  
    # True positions
    true_x = np.cos(t)
    true_y = np.sin(t)
    true_z = 0.1 * t
    
    # Derive accelerations from positions
    true_ax = -np.cos(t)  # -x for circular motion
    true_ay = -np.sin(t)  # -y for circular motion
    true_az = np.zeros_like(t) + 0.1  # constant z velocity
    
    # Create simulated IMU readings (with noise)
    imu_data_stream = []
    for i in range(num_steps):
        # Create data for 9 IMUs with varying noise levels
        imu_data = np.zeros((9, 3))
        for j in range(9):
            # Each IMU has different position and noise profile
            noise_level = 0.05 + (j * 0.01)  # Different noise level for each IMU
            imu_data[j, 0] = true_ax[i] + np.random.normal(0, noise_level)
            imu_data[j, 1] = true_ay[i] + np.random.normal(0, noise_level)
            imu_data[j, 2] = true_az[i] + np.random.normal(0, noise_level)
            
        imu_data_stream.append(imu_data)
    
    # Process the simulated data
    estimated_positions = track_position_with_imus(imu_data_stream, dt=0.1)
    
    print("Tracking complete. Estimated positions shape:", estimated_positions.shape)
    
    # Optionally, compare with true positions
    error = np.mean(np.sqrt((estimated_positions[:, 0] - true_x)**2 + 
                           (estimated_positions[:, 1] - true_y)**2 + 
                           (estimated_positions[:, 2] - true_z)**2))
    print(f"Average position error: {error:.4f} units")
