close all
clear all
clc
warning off all

clases = input("No. de clases:");
rep = input("No. de representantes por clase:");
dispX = input("¿Cuanta dispersion quieres en x?");
dispY = input("¿Cuanta dispersion quieres en y?");

media = zeros(1, 2.*clases);
A = zeros(2,rep,clases);


for i = 1:clases
    
    cenX = input("Centro X");
    cenY = input("Centro Y");
    
    cx = (randn(1,rep)+cenX)*dispX;
    cy = (randn(1,rep)+cenY)*dispY;

    w = vertcat(cx,cy);
    A(:,:,i)= w;
    media(i) = mean(cx);
    media(i+clases) = mean(cy);
    
    figure(1)
    plot(cx(1,:),cy(1,:),'.','MarkerSize',25,'DisplayName', strcat('Clase ',num2str(i)))
   
    grid on
    hold on
    legend show
    
end
persona = 0;
bandera = 'S';
while bandera == 'S'
    persona = persona + 1;
    ux = input('Ingrese X de la nueva persona: ');
    uy = input('Ingrese Y de la nueva persona: ');
    distancia = input('¿Que criterio de desiciones quieres? \n 1.Mahalanobi \n 2.Euclideana \n 3.Maxima Probabilidad');
    
    text(ux,uy,cellstr(num2str(persona)),'FontSize',15);
    if distancia == 1 %%%%%%%%%%%%%%%%%Distancia mahalanobi
        [valor,dism] = Mahalanobi(ux,uy,media,A,clases);
        fprintf("Pertenece a la facultad: %d\n",valor);
    elseif distancia == 2          %%%%%%%%%%%%%%%%%%%%%%%%%%%Distancia euclideana
        [valor,dise]=Euclideana(ux,uy,media,clases);
        fprintf("Pertenece a la facultad: %d\n",valor);
    else
        [valor,dism] = MaximaProbabilidad(ux,uy,media,A,clases);
        fprintf("Pertenece a la facultad: %d\n",valor);
    end
    
    prompt = "\nQuisiera ingresar otro? S/N: ";
    bandera = input(prompt,"s");
end


fprintf("Matriz de confusion");
matrizConfusionMaha = zeros(clases,clases);
matrizConfusionEuclideana = zeros(clases,clases);
matrizConfusionMaxProba = zeros(clases,clases);
%pruebasPorClase = round(rep*0.1);
pruebasPorClase = round(rep);
pruebas = zeros(4,pruebasPorClase*clases);

for i = 1:clases
    for j = 1:pruebasPorClase
        pruebas(1, j+(i-1)*pruebasPorClase) = i;
        [pruebas(2, j+(i-1)*pruebasPorClase),rand]= Mahalanobi(A(1,j,i),A(2,j,i),media,A,clases);
        [pruebas(3, j+(i-1)*pruebasPorClase),rand]= Euclideana(A(1,j,i),A(2,j,i),media,clases);
        [pruebas(4, j+(i-1)*pruebasPorClase),rand]= MaximaProbabilidad(A(1,j,i),A(2,j,i),media,A,clases);
    end
end

for i=1:pruebasPorClase*clases
    matrizConfusionMaha(pruebas(1,i), pruebas(2,i)) = matrizConfusionMaha(pruebas(1,i), pruebas(2,i))+1;
    matrizConfusionEuclideana(pruebas(1,i), pruebas(3,i)) = matrizConfusionEuclideana(pruebas(1,i), pruebas(3,i))+1;
    matrizConfusionMaxProba(pruebas(1,i), pruebas(4,i)) = matrizConfusionMaxProba(pruebas(1,i), pruebas(4,i))+1;
end

lineaprinmaha=0;
lineaprinEuclideana=0;
lineaprinMaxProb=0;

porclasemaha = zeros(clases);
porclaseeuclide = zeros(clases);
porclaseMaxProb = zeros(clases);

for i=1:clases
    lineaprinEuclideana = lineaprinEuclideana + matrizConfusionEuclideana(i,i);
    lineaprinmaha = lineaprinmaha + matrizConfusionMaha(i,i);
    lineaprinMaxProb = lineaprinMaxProb + matrizConfusionMaxProba(i,i);
    
    porclaseeuclide(i) = (matrizConfusionEuclideana(i,i)*100)/rep;
    porclasemaha(i)= (matrizConfusionMaha(i,i)*100)/rep;
    porclaseMaxProb(i) =(matrizConfusionMaxProba(i,i)*100)/rep;
end

acuracymaha= (lineaprinmaha*100)/(rep*clases);
acuracyeuclideana= (lineaprinEuclideana*100)/(rep*clases);
acuracyMaxProb = (lineaprinMaxProb*100)/(rep*clases);

fprintf("Matriz de confusion de Mahalanobi")
matrizConfusionMaha
fprintf("presion de ");
acuracymaha
porclasemaha

fprintf("Matriz de confusion de Euclideana")
matrizConfusionEuclideana
fprintf("con precision de");
acuracyeuclideana
porclaseeuclide

fprintf("Matriz de confusion de Maxima Probabilidad")
matrizConfisionMaxProba
fprintf("con precision de");
acuracyMaxProb
porclaseMaxProb


disp('fin de proceso...')

function [valor,distm] = Mahalanobi(ux,uy,media,A,clases)
    distm = zeros(1,clases);
    for i = 1:clases
         uxmedia = ux-media(i);
         uymedia=uy-media(i+clases);
         matrizper= horzcat(uxmedia,uymedia);
         covarianza = cov(A(:,:,i)','omitrows');
         distm(i)= matrizper*covarianza*matrizper';  
     end
    minimo=min(min(distm));
    valor=find(distm==minimo);
end


function [valor,diste] = Euclideana(ux,uy,media,clases)
    diste= zeros(1,clases);
    for i = 1:clases
        diste(i) = sqrt((ux-media(i))^2 + (uy-media(i+clases))^2);
    end
    minimo=min(min(diste));
    valor=find(diste==minimo);
end

function [valor, prob] = MaximaProbabilidad(ux,uy,media,A,clases)
    prob = zeros(1,clases);
    for i = 1:clases
         uxmedia = ux-media(i);
         uymedia = uy-media(i+clases);
         matrizper = horzcat(uxmedia,uymedia);
         covarianzatraspuesta= cov(A(:,:,i)','omitrows');
         covarianza = cov(A(:,:,i),'omitrows');
         determinante = (det(covarianza))^(1/2);
         disMaha= matrizper*covarianzatraspuesta*matrizper'; 
         prob(i) = (1/(((2*pi)^(clases/2))*determinante))* (exp((-1/2)*disMaha));
          
          
     end
   maximo =max(max(prob));
    valor = find(prob==maximo);
end