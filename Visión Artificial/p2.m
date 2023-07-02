close all
clear all
clc
warning off all

clases = input("No. de clases:");
rep = input("No. de representantes por clase:");

media = zeros(1, 2.*clases);

for i = 1:clases
	
	cenX = input("Centro X");
	cenY = input("Centro Y");
	
	cx = randn(1,rep)+cenX;
	cy = randn(1,rep)+cenY;
	
	media(i) = mean(cx);
	media(i+clases) = mean(cy);
	
	figure(1)
	plot(cx(1,:),cy(1,:),'.','MarkerSize',25,'DisplayName', strcat('Clase ',num2str(i)))
   
	grid on
	hold on
	legend show
	
end

dist = zeros(1, clases);
persona = 0;

bandera = 'S';
while bandera == 'S'
	persona = persona + 1;
	
	ux = input('Ingrese X de la nueva persona: ');
	uy = input('Ingrese Y de la nueva persona: ');
	
	%calculando distancias...
	for i = 1:clases
		dist(i) = sqrt((ux-media(i))^2 + (uy-media(i+clases))^2);
	end
	
	text(ux,uy,cellstr(num2str(persona)),'FontSize',15)
	
	%% toma de decisiones
	minimo=min(min(dist));
	valor=find(dist==minimo);

	if minimo <= 8
		fprintf("Pertenece a la facultad: %d\n",valor);
	else
		fprintf("Escuela muy lejos inscribete en una facultad\n");
	end

	prompt = "\nQuisiera ingresar otro? S/N: ";
	bandera = input(prompt,"s");

end
disp('fin de proceso...')