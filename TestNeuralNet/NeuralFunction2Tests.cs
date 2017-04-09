using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using MathNet.Numerics.LinearAlgebra;
using NeuralNet2;

namespace TestNeuralNet
{

    [TestClass]
    public class NeuralFunction2Tests
    {
        static private double sigmoid(double input)
        {
            return 1.0 / (1 + Math.Exp(-input));
        }

        static private double sigmoidDerivative(double input, double output)
        {
            return output * (1 - output);
        }

        NeuralFunction nf_1 = new NeuralFunction(1);

        NeuralFunction nf_linear =
            new NeuralFunction(2, x => x, (x, y) => 1);

        NeuralFunction nf_sigmoid
            = new NeuralFunction(2, sigmoid, sigmoidDerivative);

        DataVector vector_1 = new DataVector(Vector<double>.Build.DenseOfArray( new double[] { 1 }) );
        DataVector vector_2 = new DataVector(Vector<double>.Build.DenseOfArray( new double[] { 1, 2 }) );


        [TestMethod]
        public void CanMake()
        {
            Assert.IsNotNull(nf_1);
        }

        [TestMethod]
        public void CanMakeLinear()
        {
            Assert.IsNotNull(nf_linear);
        }

        [TestMethod]
        public void CanMakeSigmoid()
        {
            Assert.IsNotNull(nf_sigmoid);
        }

        [TestMethod]
        public void CanRun()
        {
            DataVector result = nf_1.Run(vector_1);
            Assert.AreEqual(vector_1[0], result[0]);
        }

        [TestMethod]
        public void CanRunLinear()
        {
            DataVector result = nf_linear.Run(vector_2);
            Assert.AreEqual(vector_2[0], result[0]);
            Assert.AreEqual(vector_2[1], result[1]);
        }

        [TestMethod]
        public void CanRunSigmoid()
        {
            DataVector result = nf_sigmoid.Run(vector_2);
            DataVector outputCheck = 
                new DataVector(Vector<double>.Build.DenseOfArray( new double[] { sigmoid(1), sigmoid(2) }) );
            Assert.AreEqual(outputCheck[0], result[0]);
            Assert.AreEqual(outputCheck[1], result[1]);
        }

        [TestMethod]
        public void CanBack()
        {
            DataVector result = nf_1.BackPropagate(vector_1);
            Assert.AreEqual(vector_1, result);
        }

        [TestMethod]
        public void CanBackLinear()
        {
            nf_linear.Run(vector_2);
            DataVector result = nf_linear.BackPropagate(vector_2);
            DataVector inputgradienttest = vector_2;
            Assert.AreEqual(inputgradienttest, result);
        }

        [TestMethod]
        public void CanBackSigmoid()
        {
            DataVector result = nf_sigmoid.Run(vector_2);
            DataVector inputgradient = nf_sigmoid.BackPropagate(vector_2);

            Vector<double> checkVector = Vector<double>.Build.DenseOfArray(
                new double[] { sigmoidDerivative(vector_2[0], result[0]) * vector_2[0],
                    sigmoidDerivative(vector_2[1], result[1]) * vector_2[1] }
                );
            
            DataVector gradientCheck = new DataVector(checkVector);
            Assert.AreEqual(gradientCheck, inputgradient);
        }
    }
}
