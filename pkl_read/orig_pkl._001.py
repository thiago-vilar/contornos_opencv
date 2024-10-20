import pickle
import matplotlib.pyplot as plt
import numpy as np

def main():
    filename = input("Digite o caminho para o arquivo de assinatura do contorno (.pkl): ")
    try:

        with open(filename, 'rb') as f:
            assinatura = pickle.load(f)

        # if contour.ndim == 3 and contour.shape[1] == 1:
        #     contour = contour.squeeze(axis=1)  # Ajustar para (N, 2)

        if not isinstance(assinatura, np.ndarray):
            assinatura = np.array(assinatura)

      
        if assinatura.ndim != 2 or assinatura.shape[1] != 2:
            print("Dados inesperados. Esperado um array de coordenadas com shape (N, 2).")
            return

        x, y = assinatura[:, 0], assinatura[:, 1]

        # Plotar o contorno usando scatter para garantir que os pontos não se conectem
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c='blue', s=5)  # Usar scatter em vez de plot com '-o' para evitar conectar pontos
        plt.title('Assinatura do Contorno')
        plt.xlabel('Eixo X')
        plt.ylabel('Eixo Y')
        plt.gca().invert_yaxis()  # Inverter o eixo Y para alinhamento correto da visualização
        plt.axis('equal')  # Manter a proporção dos eixos
        plt.show()

    except Exception as e:
        print(f"Ocorreu um erro ao carregar ou exibir a assinatura: {e}")

if __name__ == "__main__":
    main()
