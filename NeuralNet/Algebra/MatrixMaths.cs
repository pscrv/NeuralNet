using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public static class MatrixMaths
    {
        public static Vector MatrixLeftMultiplyVector(WeightsMatrix matrix, Vector vector)
        {
            Vector result= new FullVector(matrix.NumberOfOutputs);
            double[,] matrixArray = matrix.ToArray();
            for (int i = 0; i < matrix.NumberOfInputs; i++)
            {
                for (int j = 0; j < matrix.NumberOfOutputs; j++)
                {
                    result[j] += matrixArray[j, i] * vector[i];
                }
            }           

            return result;

            // Look up https://msdn.microsoft.com/en-us/library/ff963547.aspx
            // to see how to parallelize this
        }

        public static Vector MatrixLeftMultiplyVector(WeightsMatrix matrix, UnitVector vector)
        {
            Vector result = new FullVector(matrix.NumberOfOutputs);
            double[,] matrixArray = matrix.ToArray();
            for (int i = 0; i < matrix.NumberOfOutputs; i++)
            {
                result[i] = matrixArray[i, vector.Index];
            }
            return result;
        }
    }
}
