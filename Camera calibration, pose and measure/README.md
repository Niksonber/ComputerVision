Utiliza:
    Python 3.5.2
    OpenCV 4.1.0
    numpy
    sys
    glob

RODANDO:
---------------------------------------------------------------------

        python3 main.py
	
    OBS: Imagens de calibração obtidas em https://github.com/opencv/opencv/tree/master/samples/data

    OBS : Ao rodar o programa, o sistema vai começar a calibração (com 2 janelasm semi-abertas, sem exibir imagem), após a conclusão, vai mostrat as imagens "raw" e "undistorced", para todas as imagens de calibração (padra mudar a imagem e segir adiante basta clicar qualquer botão).(se as imagens forem grandes talvez seja preciso arrastar a imagem para vera outra).
   
    OBS: Para medir a distância em pixels basta clicar em 2 pontos da imagem, onde será desenhada uma linha

    OBS: Após a exibição das imagens de calibração, começa o RESQUISITO 3 (PARA DIFERENCIAR as imagens são exibidas com os QUADRADOS MARCADOS), da mesma forma exibindo as imagens de forma sequencial (clique qualquer botão para seguir em frente). No terminal será exibida a distancia média e desvio padrao para cada conjunto de imagens.
 
   OBS: Logo após a finalização das imagens para se ober a ,matrix extrinsica, começa o requisito 4, onde VIA TERMINAL, se pede ao usuario que INFORME qual das 3 distâncias calibradas será utilizada.
    
    OBS : As imagens de calibração para sua câmera devem estar na pasta "data/calibrate" com o formato JPG.
	  As imagens para o requisito 3 para sua câmera devem estar na pasta data/trans/"i", onde "i" é um número entre 0-2, dizendo a qual distância pertence aquelaimagem com o formato JPG.
	  As imagens para o requisito 4, onde devem estar na mesma ordem que na pasta 3, mas na pasta data/test/"i"

    OBS : Apenas as MÉDIAS dos PARÂMETROS do requisito 2 são SALVOS em xml, na própria pasta src.

