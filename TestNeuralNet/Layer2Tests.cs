using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using MathNet.Numerics.LinearAlgebra;
using NeuralNet2;

namespace TestNeuralNet
{
    [TestClass]
    public class LayerTests
    {
        [TestMethod]
        public void CanMake()
        {
            try
            {
                Layer layer1 = new Layer(
                    new WeightsMatrix(
                        Matrix<double>.Build.DenseOfArray(new double[,] { { 1 } })
                        )
                    );

                Layer layer2 = new Layer(
                    new WeightsMatrix(
                        Matrix<double>.Build.DenseOfArray(new double[,] { { 1 } })
                        ),
                    new BiasesVector(1),
                    x => 1,
                    (x, y) => 0
                    );
            }
            catch (Exception e)
            {
                Assert.Fail("Constructor threw an exception: " + e.Message);
            }

        }


        [TestMethod]
        public void CanRun()
        {
            Layer layer = new Layer(
                new WeightsMatrix(
                    Matrix<double>.Build.DenseOfArray(new double[,] { { 1 } })
                    )
                );
            
            DataVector zeroVector = new DataVector(1);

            VectorBatch result = layer.Run(zeroVector);
            Assert.AreEqual(zeroVector, result);
        }


        [TestMethod]
        public void CorrectRun()
        {
            Layer layer = new Layer(
                new WeightsMatrix(
                    Matrix<double>.Build.DenseOfArray(new double[,] { { 2, 3, 5 } })
                    )
                );

            VectorBatch input = new VectorBatch(
                Matrix<double>.Build.DenseOfArray(new double[,] { { 7, 11, 13 } })
                );
            VectorBatch outputcheck = new VectorBatch(
                Matrix<double>.Build.DenseOfArray (new double[,] { { 112 } })
                );
            
            VectorBatch result = layer.Run(input);
            Assert.AreEqual(outputcheck, result);
        }

        [TestMethod]
        public void LogisticCorrectRun()
        {
            Layer layer = new Layer(
                new WeightsMatrix(
                    Matrix<double>.Build.DenseOfArray(new double[,] { { 2, 3, 5 } })
                    ),
                new BiasesVector(1),
                NeuralFunction.__Logistic,
                NeuralFunction.__LogisticDerivative
                );

            VectorBatch input = new VectorBatch(
                Matrix<double>.Build.DenseOfArray(new double[,] { { 7, 11, 13 } })
                );

            VectorBatch outputcheck = new VectorBatch(
                Matrix<double>.Build.DenseOfArray(new double[,] { { NeuralFunction.__Logistic(112) } })
                );

            VectorBatch result = layer.Run(input);
            Assert.AreEqual(outputcheck, result);
        }

        [TestMethod]
        public void InputGradient()
        {
            Layer layer = new Layer(
                new WeightsMatrix(
                    Matrix<double>.Build.DenseOfArray(new double[,] { { 1 } })
                    )
                );

            DataVector zeroVector = new DataVector(1);
            VectorBatch result = layer.Run(zeroVector);

            DataVector oneVector = new DataVector(
                Vector<double>.Build.DenseOfArray(new double[] { 1 })
                );

            VectorBatch inputGradient = layer.BackPropagate(oneVector);
            DataVector inputGradientCheck = oneVector;

            Assert.AreEqual(inputGradientCheck, inputGradient);
        }

        [TestMethod]
        public void LogisticInputGradient()
        {
            Layer layer = new Layer(
                new WeightsMatrix(
                    Matrix<double>.Build.DenseOfArray(new double[,] { { 1 } })
                    ),
                new BiasesVector(1),
                NeuralFunction.__Logistic,
                NeuralFunction.__LogisticDerivative
                );

            DataVector zeroVector = new DataVector(1);
            VectorBatch result = layer.Run(zeroVector);

            DataVector oneVector = new DataVector(
                Vector<double>.Build.DenseOfArray(new double[] { 1 })
                );

            VectorBatch inputGradient = layer.BackPropagate(oneVector);

            DataVector inputGradientCheck = new DataVector(
                Vector<double>.Build.DenseOfArray(
                    new double[] { NeuralFunction.__LogisticDerivative(0, NeuralFunction.__Logistic(0) )})
                );

            Assert.AreEqual(inputGradientCheck, inputGradient);
        }

  

    }
}
