close all
clear all
clc
warning off all

fprintf('[ Generando archivo ]\n');
cHeader = {'R' 'G' 'B'};
commaHeader = [cHeader;repmat({','},1,numel(cHeader))];
commaHeader = commaHeader(:)';
textHeader = cell2mat(commaHeader);
archivo = 'Strawberry.csv';
fid = fopen(archivo,'w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);
disp('[ Archivo inicializado con header ]');

folder = './fruits/Strawberry/';

folders = ['./fruits/Cantaloupe/' './fruits/Granadilla/' './fruits/Mango/' './fruits/Raspberry/' './fruits/Strawberry/'];

disp('[ Obteniendo caracteristicas ]');
for i = 1:164
    fprintf('[ Caracterizando imagen %d ]\n', i);
    path = strcat(folder, num2str(i), '.jpg');
    
    img = imread(path);

    canal_rojo = img(:, :, 1);
    canal_verde = img(:, :, 2);
    canal_azul = img(:, :, 3);

    lista_rojo = reshape(canal_rojo, [], 1);
    lista_verde = reshape(canal_verde, [], 1);
    lista_azul = reshape(canal_azul, [], 1);
    lista = [lista_rojo lista_verde lista_azul];

    S = sum(lista, 2);
    S = S/3;

    for j = 1:1000
        if S(j) < 240
            dlmwrite(archivo, lista(j, :), 'delimiter', ',', '-append');
        end
    end

end
fprintf('[ Guardado de caracteristicas finalizado ]\n');
