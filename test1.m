% ----------- Load Audio ------------
[audioIn, fs] = audioread('test.wav');
audioIn = audioIn(:,1); % Use mono
audioIn = audioIn / max(abs(audioIn)); % Normalize

% ----------- Plot Waveform ------------
t = (0:length(audioIn)-1)/fs;
figure;
plot(t, audioIn);
xlabel('Time (s)');
ylabel('Amplitude');
title('Audio Waveform');

% ----------- Spectrogram ------------
windowLength = 1024;
hopLength = 512;
figure;
spectrogram(audioIn, windowLength, windowLength - hopLength, [], fs, 'yaxis');
title('Spectrogram');

% ----------- MFCC Extraction (Updated for New MATLAB) ------------
win = hamming(windowLength, 'periodic');
coeffs = mfcc(audioIn, fs, ...
    'LogEnergy', 'Ignore', ...
    'NumCoeffs', 15, ...
    'Window', win, ...
    'OverlapLength', windowLength - hopLength);

% Plot MFCCs
figure;
imagesc(coeffs');
axis xy;
xlabel('Frame Index');
ylabel('MFCC Coefficient');
colorbar;
title('MFCC Coefficients');

% Mean MFCC (for ML)
mfcc_mean = mean(coeffs, 1);

% ----------- DWT Feature Extraction ------------
level = 3;
wavelet = 'db4';
[c, l] = wavedec(audioIn, level, wavelet);

% Extract features from DWT
features_dwt = [];
for i = 1:level
    d = detcoef(c, l, i);
    energy = sum(d.^2);
    std_dev = std(d);
    p = abs(d) / sum(abs(d)) + 1e-12;
    ent = -sum(p .* log(p));
    features_dwt = [features_dwt energy std_dev ent];
end

% Include approximation coefficients at last level
a = appcoef(c, l, wavelet, level);
energy = sum(a.^2);
std_dev = std(a);
p = abs(a) / sum(abs(a)) + 1e-12;
ent = -sum(p .* log(p));
features_dwt = [features_dwt energy std_dev ent];

% ----------- Cepstrum Extraction ------------
frame = audioIn(1:windowLength); % Take 1 frame
spectrum = abs(fft(frame));
log_spectrum = log(spectrum + eps);
cepstrum = real(ifft(log_spectrum));
cepstral_coeffs = cepstrum(1:20); % First 20 coeffs

% Plot Cepstrum
figure;
plot(cepstral_coeffs);
xlabel('Quefrency');
ylabel('Amplitude');
title('Cepstral Coefficients');

% ----------- Combine Features ------------
combined_features = [mfcc_mean, features_dwt, cepstral_coeffs];

% ----------- Output Feature Vector ------------
disp('Combined Feature Vector:');
disp(combined_features);
