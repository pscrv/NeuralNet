using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using NeuralNet;
using NeuralNet.NetComponent;

namespace TestNeuralNet
{

    [TestClass]
    public class LayerTestsB
    {
        [TestMethod]
        public void CanMakeLayer()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1, 2 }, { 3, 4 } } );
            NeuralNet.NetComponent.Layer2 layer = new NeuralNet.NetComponent.Layer2(weights);
            Assert.IsNotNull(layer);
        }

        [TestMethod]
        public void CanMakeLayerWithBiases()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1, 2 }, { 3, 4 } } );
            NetworkVector biases = new NetworkVector( new double[] { 5, 7 } );
            NeuralNet.NetComponent.Layer2 layer = new NeuralNet.NetComponent.Layer2(weights, biases);
            Assert.IsNotNull(layer);
        }

        [TestMethod]
        public void CannotMakeLayerWithMismatchedBiases()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1, 2 }, { 3, 4 } } );
            NetworkVector biases = new NetworkVector( new double[] { 5, 7, 11 } );
            try
            {
                NeuralNet.NetComponent.Layer2 layer = new NeuralNet.NetComponent.Layer2(weights, biases);
                Assert.Fail("ArgumentException expected but not thrown");
            }
            catch (ArgumentException)
            { }
        }

        [TestMethod]
        public void NeuralFunctionNotNullRequiresDerivativeNotNull()
        {

           NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1, 2, 3 }, { 3, 4, 5 } } );
            NetworkVector biases = new NetworkVector(new double[] { 1, 2 });
            try
            {
                NeuralNet.NetComponent.Layer2 layer = new NeuralNet.NetComponent.Layer2(weights, biases, x => 1.0, null);
                Assert.Fail("Attempt to create Layer with non-null _neuralFunction and null _neuralFunctioinDerivative should throw and Argument exception, but did not.");
            }
            catch (ArgumentException)
            { }
        }

        [TestMethod]
        public void LayerHasRightSize()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1, 2, 3 }, { 3, 4, 5 } } );
            NeuralNet.NetComponent.Layer2 layer = new NeuralNet.NetComponent.Layer2(weights);
            Assert.AreEqual(3, layer.NumberOfInputs);
            Assert.AreEqual(2, layer.NumberOfOutputs);
        }

        [TestMethod]
        public void UnrunLayerHasZeroOutput()
        {
            NetworkMatrix weights = new NetworkMatrix(new double[,] { { 1, 2, 3 }, { 3, 4, 5 } });
            NeuralNet.NetComponent.Layer2 layer = new NeuralNet.NetComponent.Layer2(weights);

            NetworkVector outputCheck = new NetworkVector(new double[] { 0, 0 });
            Assert.AreEqual(outputCheck, layer.Output);
        }

        [TestMethod]
        public void InputGradientRequiresNonNullInput()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1 } } );
            NeuralNet.NetComponent.Layer2 layer = new NeuralNet.NetComponent.Layer2(weights);
            try
            {
                layer.InputGradient(null);
                Assert.Fail("Backpropogate should throw an ArgumentException for null input, but did not.");
            }
            catch (ArgumentException) { }
        }

        [TestMethod]
        public void InputGradientRequiresCorrectInputSize()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1 } } );
            NetworkVector badInput = new NetworkVector(new double[] { 1, 2, 3 });
            NeuralNet.NetComponent.Layer2 layer = new NeuralNet.NetComponent.Layer2(weights);
            try
            {
                layer.InputGradient(badInput);
                Assert.Fail("Backpropogate should throw an ArgumentException if input dimension is not equal to NumberOfNeuron, but did not.");
            }
            catch (ArgumentException) { }
        }        

        [TestMethod]
        public void BackpropagateRuns()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1 } } );
            NetworkVector outputgradient = new NetworkVector(new double[] { 1 });
            NeuralNet.NetComponent.Layer2 layer = new NeuralNet.NetComponent.Layer2(weights);

            NetworkVector inputGradientCheck = new NetworkVector( new double[] { 1 } );
            NetworkVector biasesGradientCheck = new NetworkVector(new double[] { 1 });
            NetworkMatrix weightsGradientCheck = new NetworkMatrix(new double[,] { { 0 } });
            Assert.AreEqual(inputGradientCheck, layer.InputGradient(outputgradient));
            Assert.AreEqual(biasesGradientCheck, layer.BiasesGradient(outputgradient));
            Assert.AreEqual(weightsGradientCheck, layer.WeightsGradient(outputgradient));
        }
    }


    [TestClass]
    public class LinearLayerTestsB
    {
        [TestMethod]
        public void LinearLayerHasRightRun()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1, 0, 1 }, { 1, 1, 0 } } );
            NetworkVector biases = new NetworkVector(new double[] { 0, 0 });
            NetworkVector inputvector = new NetworkVector(new double[] { 1, 2, 3 });
            NeuralNet.NetComponent.Layer2 layer = NeuralNet.NetComponent.Layer2.CreateLinearLayer(weights, biases);
            layer.Run(inputvector);

            NetworkVector result = layer.Output;
            NetworkVector expectedResult = new NetworkVector( new double[] { 4, 3 } );
            Assert.AreEqual(expectedResult, result);
        }
        
        [TestMethod]
        public void LinearLayerWithBiasesHasRightRun()
        {
            NetworkMatrix weights = new NetworkMatrix(new double[,] { { 1, 0, 1 }, { 1, 1, 0 } });
            NetworkVector biases = new NetworkVector(new double[] { 4, 3 });
            NetworkVector inputvector = new NetworkVector(new double[] { 1, 2, 3 });
            Layer2 layer = Layer2.CreateLinearLayer(weights, biases);
            layer.Run(inputvector);
            
            NetworkVector expectedResult = new NetworkVector( new double[] { 8, 6 });
            Assert.AreEqual(expectedResult, layer.Output);
        }

        [TestMethod]
        public void CanUseBigLinearLayer()
        {
            double[,] matrix = new double[2000, 1000];
            double[] input = new double[1000];

            for (int i = 0; i < 1000; i++)
            {
                matrix[i, i] = 1.0;
                input[i] = (double)i;
            }

            NetworkMatrix weights = new NetworkMatrix(matrix);
            NetworkVector inputvector = new NetworkVector(input);
            Layer2 layer = Layer2.CreateLinearLayer(weights);
            layer.Run(inputvector);
            double[] result = layer.Output.ToArray();

            for (int i = 0, j = 1000; i < 1000; i++, j++)
            {
                Assert.AreEqual((double)i, result[i], "Failed for i = " + i);
                Assert.AreEqual(0.0, result[j], "Failed for j = " + j);
            }
        }
        
        [TestMethod]
        public void InputGradientRunsWithZeroLayerInput()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1 } } );
            NetworkVector outputgradient = new NetworkVector(new double[] { 1 });
            Layer2 layer = Layer2.CreateLinearLayer(weights);

            NetworkVector inputGradientCheck = new NetworkVector( new double[] { 1 } );
            Assert.AreEqual(inputGradientCheck, layer.InputGradient(outputgradient));
        }

        [TestMethod]
        public void BackpropagateRunsWithNonzeroLayerInput()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1 } } );
            Layer2 layer = Layer2.CreateLinearLayer(weights);

            NetworkVector layerinput = new NetworkVector(new double[] { 2 });
            layer.Run(layerinput);

            NetworkVector outputgradient = new NetworkVector(new double[] { 1 });

            NetworkVector inputGradientCheck = new NetworkVector( new double[] { 1 } );
            Assert.AreEqual(inputGradientCheck, layer.InputGradient(outputgradient));
        }
        
        [TestMethod]
        public void InputGradientRunsTwoByThree()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1, 2, 3 }, { 2, 3, 4 } } );
            Layer2 layer = Layer2.CreateLinearLayer(weights);

            NetworkVector layerinput = new NetworkVector(new double[] { 1, 0, -1 });
            layer.Run(layerinput);

            NetworkVector outputgradient = new NetworkVector(new double[] { 1, 1 });

            NetworkVector inputGradientCheck = new NetworkVector( new double[] { 3, 5, 7 } );
            Assert.AreEqual(inputGradientCheck, layer.InputGradient(outputgradient));
            
        }

        [TestMethod]
        public void BackPropagationIsCorrect()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1, 2 }, { 3, 5 } } );
            Layer2 layer = Layer2.CreateLinearLayer(weights);

            NetworkVector layerinput = new NetworkVector(new double[] { 1, -1 });
            layer.Run(layerinput);

            NetworkVector outputgradient = new NetworkVector(new double[] { 7, 11 });

            NetworkMatrix weightsGradientCheck = new NetworkMatrix( new double[,] { { 7, -7 }, { 11, -11 } } );
            Assert.AreEqual(weightsGradientCheck, layer.WeightsGradient(outputgradient));

            NetworkVector biasesGradientCheck = new NetworkVector( new double[] { 7, 11 } );
            Assert.AreEqual(biasesGradientCheck, layer.BiasesGradient(outputgradient));

            NetworkVector inputGradientCheck = new NetworkVector( new double[] { 40, 69 } );
            Assert.AreEqual(inputGradientCheck, layer.InputGradient(outputgradient));
        }

    }



    [TestClass]
    public class SigmoidLayerTestsB
    {
        ActivationFunction logistic = NeuralNet.NetComponent.NeuralFunction.__Logistic;


        [TestMethod]
        public void CanMakeSigmoidLayer()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1, 2 }, { 3, 4 } } );
            Layer2 layer = Layer2.CreateLogisticLayer(weights);
            Assert.IsNotNull(layer);
        }

        [TestMethod]
        public void SigmoidLayerHasRightRun()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1, 0, 1 }, { 1, 1, 0 } } );
            NetworkVector inputvector = new NetworkVector(new double[] { 1, 2, 3 });
            Layer2 layer = Layer2.CreateLogisticLayer(weights);
            layer.Run(inputvector);

            NetworkVector expectedResult = new NetworkVector( new double[] { logistic(4), logistic(3) } );
            Assert.AreEqual(expectedResult, layer.Output);
        }

        [TestMethod]
        public void SigmoidLayerWithBiasesHasRightRun()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1, 0, 1 }, { 1, 1, 0 } } );
            NetworkVector biases = new NetworkVector( new double[] { 4, 3 } );
            NetworkVector inputvector = new NetworkVector(new double[] { 1, 2, 3 });
            Layer2 layer = Layer2.CreateLogisticLayer(weights, biases);
            layer.Run(inputvector);
            
            NetworkVector expectedResult = new NetworkVector( new double[] { logistic(8), logistic(6) } );
            Assert.AreEqual(expectedResult, layer.Output);
        }

        [TestMethod]
        public void CanUseBigSigmoidLayer()
        {
            double[,] weights = new double[2000, 1000];
            double[] input = new double[1000];

            for (int i = 0; i < 1000; i++)
            {
                weights[i, i] = 1.0;
                input[i] = (double)i;
            }

            NetworkVector inputvector = new NetworkVector(input);
            Layer2 layer = Layer2.CreateLogisticLayer(new NetworkMatrix(weights));
            layer.Run(inputvector);

            double[] result = layer.Output.ToArray();
            double sig0 = logistic(0.0);
            for (int i = 0, j = 1000; i < 1000; i++, j++)
            {
                Assert.AreEqual(logistic((double)i), result[i], "Failed for i = " + i);
                Assert.AreEqual(sig0, result[j], "Failed for j = " + j);
            }
        }

        [TestMethod]
        public void InputGradientRuns()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1 } } );
            NetworkVector outputgradient = new NetworkVector(new double[] { 1 });
            Layer2 layer = Layer2.CreateLogisticLayer(weights);

            NetworkVector inputGradientCheck = new NetworkVector( new double[] { 0 } );
            Assert.AreEqual(inputGradientCheck, layer.InputGradient(outputgradient));
        }

        [TestMethod]
        public void InputGradientRunsTwoByThree()
        {
            NetworkMatrix weights = new NetworkMatrix( new double[,] { { 1, 2, 3 }, { 2, 3, 4 } } );
            Layer2 layer = Layer2.CreateLogisticLayer(weights);

            NetworkVector layerinput = new NetworkVector(new double[] { 1, 0, -1 });
            layer.Run(layerinput);

            NetworkVector outputgradient = new NetworkVector(new double[] { 1, 1 });

            NetworkVector inputGradientCheck = new NetworkVector(
                new double[] { 0.31498075621051952, 0.52496792701753248, 0.7349550978245456 }
                );

            Assert.AreEqual(inputGradientCheck, layer.InputGradient(outputgradient));            
        }
    }




    [TestClass]
    public class LayerTestsA
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
            Assert.AreEqual(2, layer.NumberOfOutputs);
        }

        [TestMethod]
        public void UnrunLayerHasZeroOutput()
        {
            double[,] weights = new double[,] { { 1, 2, 3 }, { 3, 4, 5 } };
            Layer layer = new Layer(weights);
            double[] values = layer.Output.ToArray();
            Assert.AreEqual(0, values[0]);
            Assert.AreEqual(0, values[1]);
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
            NetworkVector badInput = new NetworkVector( new double[] { 1, 2, 3 } );
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
            NetworkVector outputgradient = new NetworkVector( new double[] { 1 } );
            Layer layer = new Layer(weights);
            layer.BackPropagate(outputgradient);

            double[] inputGradientCheck = new double[] { 1 };
            double[] inputGradientValues = layer.InputGradient.ToArray();
            for (int i = 0; i < layer.NumberOfInputs; i++)
            {
                Assert.AreEqual(inputGradientCheck[i], inputGradientValues[i]);
            }
        }
    }


    [TestClass]
    public class LinearLayerTestsA
    {
        [TestMethod]
        public void LinearLayerHasRightRun()
        {
            double[,] weights = new double[,] { { 1, 0, 1 }, { 1, 1, 0 } };
            NetworkVector inputvector = new NetworkVector( new double[] { 1, 2, 3 } );
            Layer layer = new LinearLayer(weights);
            layer.Run(inputvector);
            double[] result = layer.Output.ToArray();
            double[] expectedResult = new double[] { 4, 3 };
            Assert.AreEqual(expectedResult[0], result[0]);
            Assert.AreEqual(expectedResult[1], result[1]);
        }

        [TestMethod]
        public void LinearLayerWithBiasesHasRightRun()
        {
            double[,] weights = new double[,] { { 1, 0, 1 }, { 1, 1, 0 } };
            double[] biases = new double[] { 4, 3 };
            NetworkVector inputvector = new NetworkVector( new double[] { 1, 2, 3 } );
            Layer layer = new LinearLayer(weights, biases);
            layer.Run(inputvector);
            double[] result = layer.Output.ToArray();
            double[] expectedResult = new double[] { 8, 6 };
            Assert.AreEqual(expectedResult[0], result[0]);
            Assert.AreEqual(expectedResult[1], result[1]);
        }
        
        [TestMethod]
        public void CanUseBigLinearLayer()
        {
            double[,] weights = new double[2000, 1000];
            double[] input = new double[1000];

            for (int i = 0; i < 1000; i++)
            {
                weights[i, i] = 1.0;
                input[i] = (double)i;
            }

            NetworkVector inputvector = new NetworkVector(input);
            Layer layer = new LinearLayer(weights);
            layer.Run(inputvector);
            double[] result = layer.Output.ToArray();

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
            NetworkVector outputgradient = new NetworkVector( new double[] { 1 } );
            Layer layer = new LinearLayer(weights);
            layer.BackPropagate(outputgradient);

            double[] inputGradientCheck = new double[] { 1 };
            double[] inputGradientValues = layer.InputGradient.ToArray();
            for (int i  = 0; i < layer.NumberOfInputs; i++)
            {
                Assert.AreEqual(inputGradientCheck[i], inputGradientValues[i]);
            }
        }

        [TestMethod]
        public void BackpropagateRunsWithNonzeroLayerInput()
        {
            double[,] weights = new double[,] { { 1 } };
            Layer layer = new LinearLayer(weights);

            NetworkVector layerinput = new NetworkVector( new double[] { 2 } );
            layer.Run(layerinput);

            NetworkVector outputgradient = new NetworkVector( new double[] { 1 } );
            layer.BackPropagate(outputgradient);

            double[] inputGradientCheck = new double[] { 1 };
            double[] inputGradientValues = layer.InputGradient.ToArray();
            for (int i = 0; i < layer.NumberOfInputs; i++)
            {
                Assert.AreEqual(inputGradientCheck[i], inputGradientValues[i]);
            }
        }

        [TestMethod]
        public void BackpropagateRunsTwoByThree()
        {
            double[,] weights = new double[,] { { 1, 2, 3 }, { 2, 3, 4 } };
            Layer layer = new LinearLayer(weights);

            NetworkVector layerinput = new NetworkVector( new double[] { 1, 0, -1 } );
            layer.Run(layerinput);

            NetworkVector outputgradient = new NetworkVector( new double[] { 1, 1 });
            layer.BackPropagate(outputgradient);

            double[] inputGradientCheck = new double[] { 3, 5, 7 };
            double[] inputGradientValues = layer.InputGradient.ToArray();
            for (int i = 0; i < layer.NumberOfInputs; i++)
            {
                Assert.AreEqual(inputGradientCheck[i], inputGradientValues[i], string.Format("Failure for input {0}", i));
            }
        }

        [TestMethod]
        public void BackPropagateIsCorrect()
        {

            double[,] weights = new double[,] { { 1, 2}, { 3, 5} };
            Layer layer = new LinearLayer(weights);

            NetworkVector layerinput = new NetworkVector( new double[] { 1, -1 } );
            layer.Run(layerinput);

            NetworkVector outputgradient = new NetworkVector( new double[] { 7, 11 } );
            layer.BackPropagate(outputgradient);

            double[,] weightsCheck = new double[,] { { -6, 9 }, { -8, 16 } };
            LayerState state = layer.State;
            for (int i = 0; i < layer.NumberOfInputs; i++)
            {
                for (int j = 0; j < layer.NumberOfInputs; j++)
                {
                    Assert.AreEqual(weightsCheck[i, j], state.Weights[i, j], string.Format("Failed for (i, j) = ({0}, {1}", i, j));
                }
            }

            double[] biasesCheck = new double[] { -7, -11 };
            for (int i = 0; i < layer.NumberOfInputs; i++)
            {
                Assert.AreEqual(biasesCheck[i], layer.State.Biases[i]);
            }

            double[] inputGradientCheck = new double[] { 40, 69 };
            double[] inputGradientValues = layer.InputGradient.ToArray();
            for (int i = 0; i < layer.NumberOfInputs; i++)
            {
                Assert.AreEqual(inputGradientCheck[i], inputGradientValues[i], string.Format("Failure for input {0}", i));
            }
        }

    }



    [TestClass]
    public class SigmoidLayerTestsA
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
            NetworkVector inputvector = new NetworkVector( new double[] { 1, 2, 3 } );
            Layer layer = new SigmoidLayer(weights);
            layer.Run(inputvector);
            double[] result = layer.Output.ToArray();
            double[] expectedResult = new double[] { sigmoid(4), sigmoid(3) };
            Assert.AreEqual(expectedResult[0], result[0]);
            Assert.AreEqual(expectedResult[1], result[1]);
        }

        [TestMethod]
        public void SigmoidLayerWithBiasesHasRightRun()
        {
            double[,] weights = new double[,] { { 1, 0, 1 }, { 1, 1, 0 } };
            double[] biases = new double[] { 4, 3 };
            NetworkVector inputvector = new NetworkVector( new double[] { 1, 2, 3 } );
            Layer layer = new SigmoidLayer(weights, biases);
            layer.Run(inputvector);
            double[] result = layer.Output.ToArray();
            double[] expectedResult = new double[] { sigmoid(8), sigmoid(6) };
            Assert.AreEqual(expectedResult[0], result[0]);
            Assert.AreEqual(expectedResult[1], result[1]);
        }
        
        [TestMethod]
        public void CanUseBigSigmoidLayer()
        {
            double[,] weights = new double[2000, 1000];
            double[] input = new double[1000];

            for (int i = 0; i < 1000; i++)
            {
                weights[i, i] = 1.0;
                input[i] = (double)i;
            }

            NetworkVector inputvector = new NetworkVector(input);
            Layer layer = new SigmoidLayer(weights);
            layer.Run(inputvector);
            double[] result = layer.Output.ToArray();

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
            NetworkVector outputgradient = new NetworkVector( new double[] { 1 } );
            Layer layer = new SigmoidLayer(weights);
            layer.BackPropagate(outputgradient);

            double[] inputGradientCheck = new double[] { 0 };
            double[] inputGradientValues = layer.InputGradient.ToArray();
            for (int i = 0; i < layer.NumberOfInputs; i++)
            {
                Assert.AreEqual(inputGradientCheck[i], inputGradientValues[i]);
            }
        }
        [TestMethod]
        public void BackpropagateRunsTwoByThree()
        {
            double[,] weights = new double[,] { { 1, 2, 3 }, { 2, 3, 4 } };
            Layer layer = new SigmoidLayer(weights);

            NetworkVector layerinput = new NetworkVector( new double[] { 1, 0, -1 } );
            layer.Run(layerinput);

            NetworkVector outputgradient = new NetworkVector( new double[] { 1, 1 } );
            layer.BackPropagate(outputgradient);

            double[] inputGradientCheck = new double[] { 0.31498075621051952, 0.52496792701753248, 0.7349550978245456 };
            double[] inputGradientValues = layer.InputGradient.ToArray();
            for (int i = 0; i < layer.NumberOfInputs; i++)
            {
                Assert.AreEqual(inputGradientCheck[i], inputGradientValues[i], string.Format("Failure for input {0}", i));
            }
        }
    }
}
