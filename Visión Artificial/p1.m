close all
clear all
clc
warning off all

C1 = [1,2,1,4,3; 2,0,3,1,2];
C2 = [3,3,5,6,5; 4,7,2,5,0];
C3 = [7,6,9,8,7; 3,9,0,1,2];
 
media1=mean(C1,2);
media2=mean(C2,2);
media3=mean(C3,2);

bandera = 'S';
while bandera == 'S'
    ux = input('ingrese la posicion en x \n');
    uy = input('ingrese la posicion en y \n');
    y =[ux;uy];
    
    %calculando distancias...
    dist1=norm(y-media1);
    dist2=norm(y-media2);
    dist3=norm(y-media3);
    
    figure(1)
    plot(C1(1,:),C1(2,:),'ro','MarkerSize',10,'MarkerFaceColor','r')
    grid on
    hold on
    plot(C2(1,:),C2(2,:),'bo','MarkerSize',10,'MarkerFaceColor','b')
    grid on
    plot(C3(1,:),C3(2,:),'go','MarkerSize',10,'MarkerFaceColor','g')
    plot(y(1,:),y(2,:),'k*','MarkerSize',10,'MarkerFaceColor','k')
    legend('ISC','LCD','IIA','persona a decidir')
    
    %% toma de decisiones
    dist_total=[dist1 dist2 dist3]
    minimo=min(min(dist_total));
    valor=find(dist_total==minimo);

    if minimo <= 8
        fprintf("Pertenece a la facultad: %d\n",valor);
         
    else
        fprintf("Escuela muy lejos inscribete en una facultad\n");
    end

    prompt = "\nQuisiera ingresar otro? S/N: ";
    bandera = input(prompt,"s");

end
disp('fin de proceso...')