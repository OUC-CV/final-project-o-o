% 读取 HDR 和 LDR 图像
hdrImage = hdrread('data/input_images/input_images/input_hdr/doll.hdr');
ldrImage = imread('outputs/twt/doll.png');

% 调用 TMQI 函数进行图像质量评估
[Q, S, N, s_maps, s_local] = TMQI(hdrImage, ldrImage);

% 输出评估结果
fprintf('S: %.4f\n', S);
fprintf('N: %.4f\n', N);
fprintf('Q: %.4f\n', Q);