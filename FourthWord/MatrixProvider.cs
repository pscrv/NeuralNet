
using System;
using NeuralNet;

namespace FourthWord
{
    public static class MatrixProvider
    {
        private static Random __randomGenerator = new Random();


        public static NetworkMatrix GetRandom(int rowcount, int columncount)
        {
            double[,] result = new double[rowcount, columncount];
            for (int i = 0; i < rowcount; i++)
                for (int j = 0; j < columncount; j++)
                    result[i, j] = __randomGenerator.NextDouble() * 0.2 + 0.4;
            return new NetworkMatrix(result);
        }

    }


}
