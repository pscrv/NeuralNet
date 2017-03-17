using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public abstract class NetworkTrainer
    {
        #region protected members
        protected LinearTwoLayerTestNetwork _network;
        protected IEnumerable<TrainingVector> _trainingData;
        protected AdaptationStrategy _strategy;
        #endregion

        #region constructor
        public NetworkTrainer(LinearTwoLayerTestNetwork network, IEnumerable<TrainingVector> trainingdata)
        {
            _network = network;
            _trainingData = trainingdata;
            _strategy = new GradientDescent(1.0);
        }
        #endregion
        
        public abstract void Train();

        #region protected methods
        protected double _errorFunction(NetworkVector result, NetworkVector expected)
        {
            NetworkVector _getSquaredErrors = NetworkVector.ApplyFunctionComponentWise(result, expected, (x, y) => (x - y) * (x - y));
            return _getSquaredErrors.SumValues() / 2;
        }
        

        protected NetworkVector _getErrorGradient(TrainingVector tv)
        {
            _network.Run(tv.Input);
            NetworkVector gradient = _network.Output.Copy();
            gradient.Subtract(tv.Target);
            return gradient;
        }
        #endregion
    }

    

    public class OnlineNetworkTrainer : NetworkTrainer
    {
        #region constructors
        public OnlineNetworkTrainer(LinearTwoLayerTestNetwork network, IEnumerable<TrainingVector> trainingdata)
            : base(network, trainingdata)
        { }
        #endregion

        #region NetworkTrainer overrides
        public override void Train()
        {
            foreach (TrainingVector tv in _trainingData)
            {
                foreach (StateGradient sg in _network.GradientBackPropagator(_getErrorGradient(tv)))
                {
                    sg.Component.Update(
                        _strategy.BiasesUpdate(sg.Biases),
                        _strategy.WeightsUpdate(sg.Weights)
                        );
                }                
            }
        }
        #endregion
    }


    public class BatchNetworkTrainer : NetworkTrainer
    {
        #region constructor
        public BatchNetworkTrainer(LinearTwoLayerTestNetwork network, IEnumerable<TrainingVector> trainingdata)
            : base(network, trainingdata)
        { }
        #endregion

        #region public methods
        public override void Train()
        {
            Dictionary<TrainableComponent, StateGradient> gradientDictionary = new Dictionary<TrainableComponent, StateGradient>();
            StateGradient workingGradient;

            foreach (TrainingVector tv in _trainingData)
            {
                foreach (StateGradient sg in _network.GradientBackPropagator(_getErrorGradient(tv)))
                {
                    if (! gradientDictionary.ContainsKey(sg.Component)) 
                        gradientDictionary[sg.Component] =
                            new StateGradient(
                                sg.Component, 
                                new NetworkMatrix(sg.Component.NumberOfOutputs, sg.Component.NumberOfInputs),
                                new NetworkVector(sg.Component.NumberOfOutputs)
                            );

                    workingGradient = gradientDictionary[sg.Component];
                    workingGradient.Biases.Add(sg.Biases);
                    workingGradient.Weights.Add(sg.Weights); 
                }
            }

            foreach (KeyValuePair<TrainableComponent, StateGradient> componentState in gradientDictionary)
            {
                TrainableComponent component = componentState.Key;
                StateGradient gradient = componentState.Value;
                
                component.Update(
                    _strategy.BiasesUpdate(gradient.Biases),
                    _strategy.WeightsUpdate(gradient.Weights)
                    );
            }
        }
        #endregion

    }
}
