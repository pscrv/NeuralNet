using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public abstract class TrainableComponent : NetComponent, ITrainable
    {
        #region ITrainable
        public abstract void BackPropagate(NetworkVector outputgradient);
        public abstract void Update(AdaptationStrategy strategy);
            // void Run(NetworkVector input); defined in NetComponent
            // NetworkVector Output { get;  } defined in NetComponent
        #endregion

        #region protected fields
        protected WeightsMatrix _weightsGradientAccumulator;
        protected NetworkVector _biasesGradientAccumulator;
        #endregion

        #region public properties
        public abstract WeightsMatrix Weights { get; }
        public abstract NetworkVector Biases { get; }        
        #endregion

        #region constructors
        public TrainableComponent(int numberofoutputs, int numberofinputs)
        {
            _weightsGradientAccumulator = new WeightsMatrix(numberofoutputs, numberofinputs);
            _biasesGradientAccumulator = new NetworkVector(numberofoutputs);
        }
        #endregion


        #region public methods
        public abstract NetworkVector BiasesGradient(NetworkVector outputgradient);
        public abstract WeightsMatrix WeightsGradient(NetworkVector outputgradient);        
        #endregion
    }
}
