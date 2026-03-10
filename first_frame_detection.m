% Set random seed for consistency
rng(032905);

% Number of sleds to detect
N = 2;
sled_colors = cell(1,N);  % Stores color information
sled_maps = cell(1,N);
rgb_means = cell(1,N);

% Select which test case
v1 = VideoReader("test3_5_1.mp4");
% v1 = VideoReader("test3_5_2.mp4");
% v1 = VideoReader("test3_5_3.mp4");
% v1 = VideoReader("test3_5_4.mp4");
% v1 = VideoReader("test3_5_5.mp4");

% Get first frame from video
frame1 = readFrame(v1);
[m, n, ~] = size(frame1);

for i = 1:N
    sled_maps{i} = zeros(m,n);
end

% Detect checkerboard and plot points
[impoints, boardsize] = detectCheckerboardPoints(rgb2gray(frame1));
figure(1)
subplot(311)
imshow(frame1);
hold on
plot(impoints(:,1), impoints(:,2), "Linestyle", "none", "Marker",".", "Color","r");

% Slice out area surrounding checkerboard to detect colors
% Determines size of subarray
ybuffer = 0.1;
xbuffer = 0.05;

% Corners of the rectangle
top_left = impoints(1, :);
y1 = max(1, floor(top_left(1) - n*ybuffer));
x1 = max(1, floor(top_left(2) - m*xbuffer));

bottom_right = impoints(end, :);
y2 = min(n, floor(bottom_right(1) + n*ybuffer));
x2 = min(m, floor(bottom_right(2) + m*xbuffer));

width = y2 - y1;
height = x2 - x1;

drawrectangle("Position",[y1, x1, width, height]);
hold off
color_patches = frame1(x1:x2, y1:y2, :);

% Apply smoothing
ker = 1/25.*ones(5,5);
color_patches = imfilter(color_patches, ker);

% Display patch
[m0, n0, ~] = size(color_patches);
subplot(312)
imshow(color_patches);

% Use EM clustering with RGB to locate color patches
K = N + 4;  % Set number of clusters to number of sleds + 4

RGB = reshape(color_patches, m0*n0, 3);

% Get normalized RGB image
R = double(color_patches(:,:,1));
G = double(color_patches(:,:,2));
B = double(color_patches(:,:,3));

color_patches_norm = (1./(R+G+B)).*double(color_patches);

% Reshape into a feature vector
rgb = reshape(color_patches_norm, m0*n0, 3);

% Convert image to HSV
color_patches_hsv = rgb2hsv(color_patches);
h = color_patches_hsv(:,:,1);
h(h > 0.5) = 1 - h(h > 0.5);
color_patches_hsv(:,:,1) = h;

% Reshape HSV image to a vector with only H and S
hs = reshape(color_patches_hsv(:,:,1:2), m0*n0, 2);

% Concatenate to form feature vector
feature_matrix = [hs rgb];

% Run algorithm
xdist = fitgmdist(double(feature_matrix), K, "RegularizationValue", 0.001);
labeled_vec = cluster(xdist, double(feature_matrix));
labeled_patches = reshape(labeled_vec, m0, n0);
centroids = xdist.mu;

% Show the clustered image
subplot(313)
imshow(labeled_patches, [])
hold on

%%
% Choose clusters with highest saturation
patch_masks = cell(1,N);
labels = 1:K;   % Manually add labels
centroids = [centroids labels'];
sat = centroids(:,2);
centroids = sortrows(centroids((sat > 0.05) & (sat < 0.95),:), -2);

for i = 1:N
    sled_colors{i} = centroids(i,1:5);
    patch_masks{i} = labeled_patches==centroids(i,6);    % Store label to display BW im
    avg_r = uint8(mean(R(patch_masks{i})));
    avg_g = uint8(mean(G(patch_masks{i})));
    avg_b = uint8(mean(B(patch_masks{i})));
    rgb_means{i} = [avg_r,avg_g,avg_b];
    plot(i*30, 1, "Marker",".", "MarkerSize", 24,"Color",[avg_r,avg_g,avg_b])
end

%% Scan for sleds using Euclidian distance

dist_thresh = 0.1;

% Get normalized RGB image
R = double(frame1(:,:,1));
G = double(frame1(:,:,2));
B = double(frame1(:,:,3));

frame1_norm = (1./(R+G+B)).*double(frame1);

% Convert image to HSV
frame1_hsv = rgb2hsv(frame1);
h = frame1_hsv(:,:,1);
h(h > 0.5) = 1 - h(h > 0.5);
frame1_hsv(:,:,1) = h;

for i = 1:m
    for j = 1:n
        rgb = squeeze(frame1(i,j,:))';

        a = squeeze(frame1_hsv(i,j,1:2));
        b = squeeze(frame1_norm(i,j,:));
        pixel_vector = [a' b'];
        
        for k = 1:N

            % Compute normalized distance using hs rgb
            d = sum((sled_colors{k} - pixel_vector).^2)/sqrt(5);
            
            if(d < dist_thresh)
                sled_maps{k}(i,j) = 1;
            end
        end
    end
end

%%
for i = 1:N
    figure(1+i)
    hold on
    imshow(sled_maps{i})
end

figure(N+2)
imshow(frame1)