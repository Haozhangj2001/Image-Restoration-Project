% read original images from the data file 
images = load('dataset.mat');
img_1 = images.I1 ./ 255;
img_2 = images.I2 ./ 255;

% PSF of the motion blur, length is 21 and direction is 11 degrees
PSF = fspecial('motion', 21, 11);

% add motion blur to the images
blurred_1 = imfilter(img_1, PSF, 'conv', 'circular');
blurred_2 = imfilter(img_2, PSF, 'conv', 'circular');

% add gaussian noise to the blurred images
noise_mean = 0;
noise_var = 0.001;
g_1 = imnoise(blurred_1, 'gaussian', noise_mean, noise_var); 
g_2 = imnoise(blurred_2, 'gaussian', noise_mean, noise_var);

% define MSE metric function
function result = MSE(f, f_e)
    [M, N] = size(f); % size of the input images
    result = sum((f.*255 - f_e.*255).^2, 'all') / (M * N);
end

% define inverse filtering function
function result = inverse_filter(input, PSF)
    input_fft = fftshift(fft2(input)); % convert the input image to spectrum
    [M, N] = size(input_fft); % size of input image spectrum
    H = fftshift(psf2otf(PSF, [M, N])); % derive the transfer function of motion blur from PSF
    result_fft = input_fft ./ H;
    result = real(ifft2(ifftshift(result_fft)));
end

% define improved inverse filtering function: option1
function result = improved_inverse_filter_1(input, PSF, radius)
    input_fft = fftshift(fft2(input)); % convert the input image to spectrum
    [M, N] = size(input_fft); % size of input image spectrum
    H = fftshift(psf2otf(PSF, [M, N])); % derive the transfer function of motion blur from PSF
    result_fft = input_fft;
    % only if frequency is lower than radius, apply inverse filtering
    for i = 1: M
        for j = 1: N
            if sqrt((i - M / 2).^2 + (j - N / 2).^2) < radius
                result_fft(i, j) = input_fft(i, j) ./ H(i, j);
            end
        end
    end
    result = real(ifft2(ifftshift(result_fft)));
end

% define improved inverse filtering function: option2
function result = improved_inverse_filter_2(input, PSF, threshhold)
    input_fft = fftshift(fft2(input)); % convert the input image to spectrum
    [M, N] = size(input_fft); % size of input image spectrum
    H = fftshift(psf2otf(PSF, [M, N])); % derive the transfer function of motion blur from PSF
    H_modified = zeros(M, N);
    % only if H(u)(v) is larger than threshhold, apply inverse filtering
    non_zeros = H > threshhold;
    H_modified(non_zeros) = 1 ./ H(non_zeros);
    result_fft = input_fft .* H_modified;
    result = real(ifft2(ifftshift(result_fft)));
end

% define wiener filtering function
function result = wiener(input, PSF, K)
    input_fft = fftshift(fft2(input)); % convert the input image to spectrum
    [M, N] = size(input_fft); % size of input image spectrum
    H = fftshift(psf2otf(PSF, [M, N])); % derive the transfer function of motion blur from PSF
    weiner_filter = conj(H) ./ (abs(H).^2 + K); % optimal weiner filter, K = 1/ SNR
    result_fft = input_fft .* weiner_filter;
    result = real(ifft2(ifftshift(result_fft)));
end

% define constrained least squares filtering (CLSF) function
function [result, gamma] = clsf(input, PSF, gamma0, noise_mean, noise_var)
    input_fft = fftshift(fft2(input)); % convert the input image to spectrum
    [M, N] = size(input_fft); % size of input image spectrum
    H = fftshift(psf2otf(PSF, [M, N])); % derive the transfer function of motion blur from PSF
    p = [0 -1 0; -1 4 -1; 0 -1 0]; % p is the laplacian
    P = fftshift(psf2otf(p, [M, N])); % Fourier transform of the laplacian
    clsf_filter = conj(H) ./ (abs(H).^2 + gamma0 *abs(P).^2); % the CLSF filter before iterating
    result_fft = input_fft .* clsf_filter;
    result = ifft2(ifftshift(result_fft));
    g_estimated = imfilter(result, PSF, 'conv', 'circular');
    % iterate to find the best gamma
    gamma = gamma0;
    error = sum((input - g_estimated).^2, 'all') - M * N * (noise_mean.^2 + noise_var);
    step_len = 0.1;
    while(abs(error) > 0.0001)
        if ( error > 0.0001)
            gamma = gamma - step_len;
        else
            gamma = gamma + step_len;
        end
        %disp(gamma);
        clsf_filter = conj(H) ./ (abs(H).^2 + gamma *abs(P).^2); % the CLSF filter before iterating
        result_fft = input_fft .* clsf_filter;
        result = real(ifft2(ifftshift(result_fft)));
        g_estimated = imfilter(result, PSF, 'conv', 'circular');
        error_new = sum((input - g_estimated).^2, 'all') - M * N * (noise_mean.^2 + noise_var);
        if (abs(error_new) >= abs(error))
            step_len = step_len * 0.95;
        end
        error = error_new;
        disp(error);
    end
end



% Plot the restored images when only motion blur is applied

% First calculate the mse of the ditorted images for comparison
mse_motion = MSE(blurred_1, img_1);
mse_motion_noise = MSE(g_1, img_1);

figure(1), plot(1);
subplot(1, 2, 1), imshow(blurred_1);
title("motion blurred image, MSE =  "+ mse_motion);
I = inverse_filter(blurred_1, PSF);
subplot(1, 2, 2), imshow(I);
mse = MSE(img_1, I);
title("inverse filtered image, MSE =  "+ mse);
sgtitle('Inverse Filtering');


figure(2), plot(2);
subplot(2, 4, 1), imshow(blurred_1);
title("motion blurred image, MSE =  "+ mse_motion);
radius = [35, 30, 25, 20, 15, 10, 5];
for i = 1: length(radius)
    I = improved_inverse_filter_1(blurred_1, PSF, radius(i));
    subplot(2, 4, i + 1), imshow(I);
    mse = MSE(img_1, I);
    title("radius = " + radius(i) + ", MSE =  "+ mse);
end
sgtitle('Improved Inverse Filtering: Solution 1');

figure(3), plot(3);
subplot(2, 4, 1), imshow(blurred_1);
title("motion blurred image, MSE =  "+ mse_motion);
threshhold = [0.5, 0.1, 0.05, 0.03, 0.01, 0.005, 0.001];
for i = 1: length(threshhold)
    I = improved_inverse_filter_2(blurred_1, PSF, threshhold(i));
    subplot(2, 4, i + 1), imshow(I);
    mse = MSE(img_1, I);
    title("threshhold = " + threshhold(i) + ", MSE = " + mse);
end
sgtitle('Improved Inverse Filtering: Solution 2');

figure(4), plot(4);
subplot(1, 2, 1), imshow(blurred_1);
title("motion blurred image, MSE =  "+ mse_motion);
K = 0; %  no noise present, weiner filter becomes inverse filter
I = wiener(blurred_1, PSF, K);
subplot(1, 2, 2), imshow(I);
mse = MSE(img_1, I);
title("weiner filtering: K = 0, MSE =  "+ mse);
sgtitle('Weiner Filtering');

figure(5), plot(5);
subplot(1, 2, 1), imshow(blurred_1);
title("motion blurred image, MSE =  "+ mse_motion);
gamma0 = 0; %  no noise present, clsf filter becomes inverse filter
[I, gamma] = clsf(blurred_1, PSF, gamma0, 0, 0);
subplot(1, 2, 2), imshow(I);
mse = MSE(img_1, I);
title("CLSF filtering: gamma = 0, MSE = " + mse);
sgtitle('CLSF Filtering');



% Plot the restored images when motion blur and gaussian noise are both applied
figure(6), plot(6);
subplot(1, 2, 1), imshow(g_1);
title("motion blurred and noisy image, MSE = " + mse_motion_noise);
I = inverse_filter(g_1, PSF);
subplot(1, 2, 2), imshow(I);
mse = MSE(img_1, I);
title("inverse filtered image, MSE = " + mse);
sgtitle('Inverse Filtering');

figure(7), plot(7);
subplot(2, 4, 1), imshow(g_1);
title("motion blurred and noisy image, MSE = " + mse_motion_noise);
radius = [25, 20, 15, 10, 7, 5, 3];
for i = 1: length(radius)
    I = improved_inverse_filter_1(g_1, PSF, radius(i));
    mse = MSE(img_1, I);
    subplot(2, 4, i + 1), imshow(I);
    title("radius = " + radius(i) + ", MSE = " + mse);
end
sgtitle('Improved Inverse Filtering: Solution 1');

figure(8), plot(8);
subplot(2, 4, 1), imshow(g_1);
title("motion blurred and noisy image, MSE = " + mse_motion_noise);
threshhold = [0.7, 0.5, 0.3, 0.1, 0.05, 0.03, 0.01];
for i = 1: length(threshhold)
    I = improved_inverse_filter_2(g_1, PSF, threshhold(i));
    mse = MSE(img_1, I);
    subplot(2, 4, i + 1), imshow(I);
    title("threshhold = " + threshhold(i) + ", MSE = " + mse);
end
sgtitle('Improved Inverse Filtering: Solution 2');

figure(9), plot(9);
subplot(2, 4, 1), imshow(g_1);
title("motion blurred and noisy image, MSE = " + mse_motion_noise);
K = [0.5, 0.1, 0.05, 0.03, 0.01, 0.005, 0.001]; 
for i = 1: length(K)
    I = wiener(g_1, PSF, K(i));
    mse = MSE(img_1, I);
    subplot(2, 4, i + 1), imshow(I);
    title("K = " + K(i) + ", MSE = " + mse);
end
sgtitle('Wiener Filtering');

figure(10), plot(10);
subplot(1, 2, 1), imshow(g_1);
title("motion blurred and noisy image, MSE = " + mse_motion_noise);
gamma0 = 100 * sqrt(noise_var); %  no noise present, clsf filter becomes inverse filter
[I, gamma] = clsf(g_1, PSF, gamma0, noise_mean, noise_var);
mse = MSE(img_1, I);
subplot(1, 2, 2), imshow(I);
title("CLSF filtering: gamma = " + gamma + ", MSE = " + mse);
sgtitle('CLSF Filtering');

% Compare the Wiener and CLSF filter in terms of noise variance (power)
delta = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05];
for i = 1: length(delta)
    figure(10 + i), plot(10 + i);
    g = imnoise(blurred_1, 'gaussian', noise_mean, delta(i));
    subplot(1, 3, 1), imshow(g);
    mse_g = MSE(g, img_1);
    title("noisy image, delta = " + delta(i) + ", MSE = " + mse_g);

    I_wiener = wiener(g, PSF, 25*delta(i));
    subplot(1, 3, 2), imshow(I_wiener);
    mse = MSE(I_wiener, img_1);
    title("Wiener, K = " + 30*delta(i) + ", MSE = " + mse);

    [I_clsf, gamma] = clsf(g, PSF,  100 * sqrt(delta(i)), noise_mean, delta(i));
    subplot(1, 3, 3), imshow(I_clsf);
    mse = MSE(I_clsf, img_1);
    title("CLSF, gamma = " + gamma + ", MSE = " + mse);
    sgtitle("Noise delta = " + delta(i));
end
