close all
clear all
clc
warning off all

fprintf('[ Generando archivo ]\n');
cHeader = {'Area' 'Perimetro' 'Metrica'};
commaHeader = [cHeader;repmat({','},1,numel(cHeader))];
commaHeader = commaHeader(:)';
textHeader = cell2mat(commaHeader);
archivo = 'Cantaloupe.csv';
fid = fopen(archivo,'w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);
disp('[ Archivo inicializado con header ]');

folder = './fruits/Cantaloupe/';
disp('[ Obteniendo caracteristicas ]');
for i = 1:164
	fprintf('[ Caracterizando imagen %d ]\n', i);
	
	path = strcat(folder, num2str(i), '.jpg');
	
	img = imread(path);
	imagen = img;
	imagen = im2double(imagen);
	umbral_blanco = 0.9;
	mascara_blanco = imagen > umbral_blanco;
	factor_oscurecimiento = 0.05;
	imagen_oscurecida = imagen;
	imagen_oscurecida(~mascara_blanco) = imagen_oscurecida(~mascara_blanco) * factor_oscurecimiento;
	imagen_oscurecida = min(max(imagen_oscurecida, 0), 1);
	imagen_gris = rgb2gray(imagen_oscurecida);
	imagen_not = imcomplement(imagen_gris);
	im_filtrada = wiener2(imagen_not,[10 10]);

	im_bin = imbinarize(im_filtrada);
	
	[m,n] = size(im_bin);
	im_bin2=imbinarize(zeros(m,n));
	for i = 1:m
		for j = 1:n
			if(im_bin(i,j) == 1)
				im_bin2(i:i+2,j:j+2) = 1;  
			end
		end
	end
	im_bin = im_bin2;
	[Bordes, Objetos] = bwboundaries(im_bin,'noholes');
	num_Objetos = length(Bordes);
	stats = regionprops(Objetos, 'Area');

	borde=Bordes{1};
	delta_sq = diff(borde).^2;
	area = stats(1).Area;
	perimetro = sum(sqrt(sum(delta_sq,2)));
	metrica = 4*pi*area/perimetro^2;

	objeto = [double(area) double(perimetro) double(metrica)];
	dlmwrite(archivo, objeto, 'delimiter', ',', '-append');
end
fprintf('[ Guardado de caracteristicas finalizado ]\n');
