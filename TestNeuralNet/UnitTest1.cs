using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;

namespace TestNeuralNet
{
    [TestClass]
    public class LayerTests
    {
        [TestMethod]
        public void CanMakeLayer()
        {
            double[,] weights = new double[,] { { 1, 2 }, { 3, 4 } };
            Layer layer = new Layer(weights);
            Assert.IsNotNull(layer);
        }

        [TestMethod]
        public void CanMakeLayerWithBiases()
        {
            double[,] weights = new double[,] { { 1, 2 }, { 3, 4 } };
            double[] biases = new double[] { 5, 7 };
            Layer layer = new Layer(weights, biases);
            Assert.IsNotNull(layer);
        }

        [TestMethod]
        public void CannotMakeLayerWithMismatchedBiases()
        {
            double[,] weights = new double[,] { { 1, 2 }, { 3, 4 } };
            double[] biases = new double[] { 5, 7, 11 };
            try
            {
                Layer layer = new Layer(weights, biases);
                Assert.Fail("ArgumentException expected but not thrown");
            }
            catch (ArgumentException)
            { }
        }

        [TestMethod]
        public void NeuralFunctionNotNullRequiresDerivativeNotNull()
        {

            double[,] weights = new double[,] { { 1, 2, 3 }, { 3, 4, 5 } };
            try
            {
                Layer layer = new Layer(weights, x => 1.0, null);
                Assert.Fail("Attempt to create Layer with non-null _neuralFunction and null _neuralFunctioinDerivative should throw and Argument exception, but did not.");
            }
            catch (ArgumentException)
            { }
        }
        
        [TestMethod]
        public void LayerHasRightSize()
        {
            double[,] weights = new double[,] { { 1, 2, 3}, { 3, 4, 5 } };
            Layer layer = new Layer(weights);
            Assert.AreEqual(3, layer.NumberOfInputs);
            Assert.AreEqual(2, layer.NumberOfNeurons);
        }

        [TestMethod]
        public void UnrunLayerHasZeroOutput()
        {
            double[,] weights = new double[,] { { 1, 2, 3 }, { 3, 4, 5 } };
            Layer layer = new Layer(weights);
            Assert.AreEqual(0, layer.Output[0]);
            Assert.AreEqual(0, layer.Output[1]);
        }

        [TestMethod]
        public void BackpropagateRequiresNonNullInput()
        {
            double[,] weights = new double[,] { { 1 } };            
            Layer layer = new Layer(weights);
            try
            {
                layer.BackPropagate(null);
                Assert.Fail("Backpropogate should throw an ArgumentException for null input, but did not.");
            }
            catch (ArgumentException) { }
        }

        [TestMethod]
        public void BackpropagateRequiresCorrectInputSize()
        {
            double[,] weights = new double[,] { { 1 } };
            double[] badInput = new double[] { 1, 2, 3 };
            Layer layer = new Layer(weights);
            try
            {
                layer.BackPropagate(badInput);
                Assert.Fail("Backpropogate should throw an ArgumentException if input dimension is not equal to NumberOfNeuron, but did not.");
            }
            catch (ArgumentException) { }
        }

        
        [TestMethod]
        public void BackpropagateRuns()
        {
            double[,] weights = new double[,] { { 1 } };
            double[] outputgradient = new double[] { 1 };
            Layer layer = new Layer(weights);
            layer.BackPropagate(outputgradient);

            double[] inputGradientCheck = new double[] { 1 };
            for (int i = 0; i < layer.NumberOfInputs; i++)
            {
                Assert.AreEqual(inputGradientCheck[i], layer.InputGradient[i]);
            }
        }
    }


    [TestClass]
    public class LinearLayerTests
    {
        [TestMethod]
        public void LinearLayerHasRightRun()
        {
            double[,] weights = new double[,] { { 1, 0, 1 }, { 1, 1, 0 } };
            double[] inputvector = new double[] { 1, 2, 3 };
            Layer layer = new LinearLayer(weights);
            double[] result = layer.Run(inputvector);
            double[] expectedResult = new double[] { 4, 3 };
            Assert.AreEqual(expectedResult[0], result[0]);
            Assert.AreEqual(expectedResult[1], result[1]);
        }

        [TestMethod]
        public void LinearLayerWithBiasesHasRightRun()
        {
            double[,] weights = new double[,] { { 1, 0, 1 }, { 1, 1, 0 } };
            double[] biases = new double[] { 4, 3 };
            double[] inputvector = new double[] { 1, 2, 3 };
            Layer layer = new LinearLayer(weights, biases);
            double[] result = layer.Run(inputvector);
            double[] expectedResult = new double[] { 8, 6 };
            Assert.AreEqual(expectedResult[0], result[0]);
            Assert.AreEqual(expectedResult[1], result[1]);
        }
        
        [TestMethod]
        public void CanUseBigLinearLayer()
        {
            double[,] weights = new double[2000, 1000];
            double[] inputvector = new double[1000];

            for (int i = 0; i < 1000; i++)
            {
                weights[i, i] = 1.0;
                inputvector[i] = (double)i;
            }            

            Layer layer = new LinearLayer(weights);
            double[] result = layer.Run(inputvector);

            for (int i = 0, j = 1000; i < 1000; i++, j++)
            {
                Assert.AreEqual((double)i, result[i], "Failed for i = " + i);
                Assert.AreEqual(0.0, result[j],  "Failed for j = " + j);
            }
        }

        [TestMethod]
        public void BackpropagateRunsWithZeroLayerInput()
        {
            double[,] weights = new double[,] { { 1 } };
            double[] outputgradient = new double[] { 1 };
            Layer layer = new LinearLayer(weights);
            layer.BackPropagate(outputgradient);

            double[] inputGradientCheck = new double[] { 1 };
            for (int i  = 0; i < layer.NumberOfInputs; i++)
            {
                Assert.AreEqual(inputGradientCheck[i], layer.InputGradient[i]);
            }
        }

        [TestMethod]
        public void BackpropagateRunsWithNonzeroLayerInput()
        {
            double[,] weights = new double[,] { { 1 } };
            Layer layer = new LinearLayer(weights);

            double[] layerinput = new double[] { 2 };
            layer.Run(layerinput);

            double[] outputgradient = new double[] { 1 };
            layer.BackPropagate(outputgradient);

            double[] inputGradientCheck = new double[] { 1 };
            for (int i = 0; i < layer.NumberOfInputs; i++)
            {
                Assert.AreEqual(inputGradientCheck[i], layer.InputGradient[i]);
            }
        }

        [TestMethod]
        public void BackpropagateRunsTwoByThree()
        {
            double[,] weights = new double[,] { { 1, 2, 3 }, { 2, 3, 4 } };
            Layer layer = new LinearLayer(weights);

            double[] layerinput = new double[] { 1, 0, -1 };
            layer.Run(layerinput);

            double[] outputgradient = new double[] { 1, 1 };
            layer.BackPropagate(outputgradient);

            double[] inputGradientCheck = new double[] { 3, 5, 7 };
            for (int i = 0; i < layer.NumberOfInputs; i++)
            {
                Assert.AreEqual(inputGradientCheck[i], layer.InputGradient[i], string.Format("Failure for input {0}", i));
            }
        }

    }



    [TestClass]
    public class SigmoidLayerTests
    {
        private double sigmoid(double x)
        {
            return 1.0 / (1 + Math.Exp(-x));
        }


        [TestMethod]
        public void CanMakeSigmoidLayer()
        {
            double[,] weights = new double[,] { { 1, 2 }, { 3, 4 } };
            Layer layer = new SigmoidLayer(weights);
            Assert.IsNotNull(layer);
        }
        

        [TestMethod]
        public void SigmoidLayerHasRightRun()
        {
            double[,] weights = new double[,] { { 1, 0, 1 }, { 1, 1, 0 } };
            double[] inputvector = new double[] { 1, 2, 3 };
            Layer layer = new SigmoidLayer(weights);
            double[] result = layer.Run(inputvector);
            double[] expectedResult = new double[] { sigmoid(4), sigmoid(3) };
            Assert.AreEqual(expectedResult[0], result[0]);
            Assert.AreEqual(expectedResult[1], result[1]);
        }

        [TestMethod]
        public void SigmoidLayerWithBiasesHasRightRun()
        {
            double[,] weights = new double[,] { { 1, 0, 1 }, { 1, 1, 0 } };
            double[] biases = new double[] { 4, 3 };
            double[] inputvector = new double[] { 1, 2, 3 };
            Layer layer = new SigmoidLayer(weights, biases);
            double[] result = layer.Run(inputvector);
            double[] expectedResult = new double[] { sigmoid(8), sigmoid(6) };
            Assert.AreEqual(expectedResult[0], result[0]);
            Assert.AreEqual(expectedResult[1], result[1]);
        }
        
        [TestMethod]
        public void CanUseBigSigmoidLayer()
        {
            double[,] weights = new double[2000, 1000];
            double[] inputvector = new double[1000];

            for (int i = 0; i < 1000; i++)
            {
                weights[i, i] = 1.0;
                inputvector[i] = (double)i;
            }

            Layer layer = new SigmoidLayer(weights);
            double[] result = layer.Run(inputvector);

            double sig0 = sigmoid(0.0);
            for (int i = 0, j = 1000; i < 1000; i++, j++)
            {
                Assert.AreEqual(sigmoid((double)i), result[i], "Failed for i = " + i);
                Assert.AreEqual(sig0, result[j], "Failed for j = " + j);
            }
        }
        
        [TestMethod]
        public void BackpropagateRuns()
        {
            double[,] weights = new double[,] { { 1 } };
            double[] outputgradient = new double[] { 1 };
            Layer layer = new SigmoidLayer(weights);
            layer.BackPropagate(outputgradient);

            double[] inputGradientCheck = new double[] { 0 };
            for (int i = 0; i < layer.NumberOfInputs; i++)
            {
                Assert.AreEqual(inputGradientCheck[i], layer.InputGradient[i]);
            }
        }
        [TestMethod]
        public void BackpropagateRunsTwoByThree()
        {
            double[,] weights = new double[,] { { 1, 2, 3 }, { 2, 3, 4 } };
            Layer layer = new SigmoidLayer(weights);

            double[] layerinput = new double[] { 1, 0, -1 };
            layer.Run(layerinput);

            double[] outputgradient = new double[] { 1, 1 };
            layer.BackPropagate(outputgradient);

            double[] inputGradientCheck = new double[] { 0.31498075621051952, 0.52496792701753248, 0.7349550978245456 };
            for (int i = 0; i < layer.NumberOfInputs; i++)
            {
                Assert.AreEqual(inputGradientCheck[i], layer.InputGradient[i], string.Format("Failure for input {0}", i));
            }
        }
    }
}
