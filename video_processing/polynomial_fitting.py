import open3d as o3d
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Load the point cloud
pcd = o3d.io.read_point_cloud('video_processing/point_clouds/@006r 255 2024-09-12 22-59-36.pcd')

# Convert point cloud to numpy array and extract points
points = np.asarray(pcd.points)

# Assuming the points are organized as (x, y, z)
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# Combine x and z to form the input features for polynomial fitting
X = np.vstack((x, z)).T

# Fit a polynomial regression model to predict y
degree = 4
poly_features = PolynomialFeatures(degree)
X_poly = poly_features.fit_transform(X)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_poly, y)

# Predict y using the fitted model
y_pred = model.predict(X_poly)
residuals = np.abs(y - y_pred)

# Threshold for error
error_threshold = 50  # Adjust this threshold as needed

# Identify points with high error
high_error_indices = residuals > error_threshold
# Display coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
# Randomly sample 10% of the points for visualization
sample_indices = np.random.choice(points.shape[0], size=int(points.shape[0] * 0.003), replace=False)
sampled_x = x[sample_indices]
sampled_y = y[sample_indices]
sampled_z = z[sample_indices]
sampled_y_pred = y_pred[sample_indices]

# Visualize the original y and predicted y for the sampled points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sampled_x, sampled_z, sampled_y, color='red', label='Original y')
#ax.scatter(sampled_x, sampled_z, sampled_y_pred, color='blue', label='Predicted y', alpha=0.5)
# Plot high-error points
# high_error_x = x[high_error_indices]
# high_error_y = y[high_error_indices]
# high_error_z = z[high_error_indices]
# ax.scatter(high_error_x, high_error_z, high_error_y, color='green', label='High Error Points', marker='^')
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')
plt.title('Polynomial Surface Fit (Sampled Points)')
plt.legend()
plt.show()



