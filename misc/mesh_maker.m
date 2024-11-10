% Load the point cloud from a PCD file using Computer Vision Toolbox
ptCloud = pcread('point_clouds/@006r 255 2024-09-12 22-59-36.pcd');

% Downsample the point cloud using a larger grid size for less memory usage
gridSize = 0.02;  % Increase the grid size for more aggressive downsampling
ptCloudDownsampled = pcdownsample(ptCloud, 'gridAverage', gridSize);

% Extract x, y, z coordinates from the downsampled point cloud
x = double(ptCloudDownsampled.Location(:, 1));  
y = double(ptCloudDownsampled.Location(:, 2));  
z = double(ptCloudDownsampled.Location(:, 3));  

% Define grid resolution for the interpolation (adjust based on your data)
xq = linspace(min(x), max(x), 100);  % Reduced grid resolution
yq = linspace(min(y), max(y), 100);

% Create a 2D meshgrid for interpolation
[Xq, Yq] = meshgrid(xq, yq);

% Use scatteredInterpolant for smoother interpolation
F = scatteredInterpolant(x, y, z, 'natural', 'none');
Zq = F(Xq, Yq);

% Optional: Apply Gaussian smoothing to reduce jaggedness
Zq = imgaussfilt(Zq, 0.5);  % Adjust the second argument as needed

% Plot the interpolated surface
figure;
mesh(Xq, Yq, Zq);
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Downsampled Spider Web Point Cloud Surface');

% Maintain aspect ratio
% axis equal;  
axis tight;  
colormap jet;  
view(3);  % Set to 3D view