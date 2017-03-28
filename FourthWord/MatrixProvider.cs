
using System;
using NeuralNet;

namespace FourthWord
{
    public static class MatrixProvider
    {
        private static Random __randomGenerator = new Random();


        public static WeightsMatrix GetRandom(int rowcount, int columncount)
        {
            if (rowcount <= 0 || columncount <= 0)
                throw new ArgumentException("Attempt to create a matrix with zero or fewer columns or rows.");

            double scaleFactor = 1.0 / columncount;
            double[,] result = new double[rowcount, columncount];
            for (int i = 0; i < rowcount; i++)
                for (int j = 0; j < columncount; j++)
                    result[i, j] = scaleFactor * (__randomGenerator.NextDouble() * 0.6 + 0.2);
            return new WeightsMatrix(result);
        }

    }


}
