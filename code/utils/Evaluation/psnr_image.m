
function psnr_val = psnr_image(img_rec, img_true)
    mse = mean((img_rec(:) - img_true(:)).^2);
    if mse == 0
        psnr_val = Inf;
    else
        psnr_val = 10 * log10(1 / mse);
    end
end
