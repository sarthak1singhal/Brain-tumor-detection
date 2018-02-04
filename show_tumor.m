function image = show_tumor( I )

im1 = I;
I=im1;
I = imresize(I,[200,200]);
I= rgb2gray(I);
I= im2bw(I,.6);
hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(I), hy, 'replicate');
Ix = imfilter(double(I), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);
L = watershed(gradmag);
Lrgb = label2rgb(L);
se = strel('disk', 20);
Io = imopen(I, se);
Ie = imerode(I, se);
Iobr = imreconstruct(Ie, I);
Iobrd = imdilate(Iobr, se);
Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);
I2 = I;
fgm = imregionalmax(Iobrcbr);
I2(fgm) = 255;
se2 = strel(ones(5,5));
fgm2 = imclose(fgm, se2);
fgm3 = imerode(fgm2, se2);
fgm4 = bwareaopen(fgm3, 20);
I3 = I;
bw =Iobrcbr;
figure
lo = anisodiff(im1,3,1,1,1);
imshow (lo);
imshow(bw), title('only tumor')
image = bw;
end

