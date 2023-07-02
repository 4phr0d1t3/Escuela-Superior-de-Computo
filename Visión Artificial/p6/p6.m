close all
clear all
clc
warning off all

fprintf('[ Generando archivo ]\n');
cHeader = {'area' 'circularity' 'extent' 'perimeter' 'solidity'};
commaHeader = [cHeader;repmat({','},1,numel(cHeader))];
commaHeader = commaHeader(:)';
textHeader = cell2mat(commaHeader);
fid = fopen('data.csv','w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);
fprintf('[ Archivo inicializado con header ]\n');


fprintf('[ Obteniendo caracteristicas ]\n');
for i = 1:114
    if i<10
        path = strcat('./images/IMAG00', num2str(i), '.BMP');
    elseif i<100
        path = strcat('./images/IMAG0', num2str(i), '.BMP');
    else
        path = strcat('./images/IMAG', num2str(i), '.BMP');
    end

    img = imread(path);
    img = wiener2(img, [50,50]);
    img = im2bw(img,0.30);
    img = bwareaopen(img,130);
    img = imfill(img, 'holes');
    
    stats = regionprops(img,'Area','Circularity','Extent','Perimeter','Solidity');
    
    areaList = stats.Area;
    extentList = stats.Extent;
    circularityList = stats.Circularity;
    perimeterList = stats.Perimeter;
    solidityList = stats.Solidity;
    objects = [stats.Area; stats.Circularity; stats.Extent; stats.Perimeter; stats.Solidity];
    
    objects = transpose(objects);
    dlmwrite('data.csv',objects,'delimiter',',','-append');
end
fprintf('[ Guardado de caracteristicas finalizado ]\n');