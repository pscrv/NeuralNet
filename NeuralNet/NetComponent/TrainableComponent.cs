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
        public abstract void BackPropagate(NetworkVector outputgradient, NetworkVector input);
        public abstract void BackPropagate(VectorBatch outputgradient);
        public abstract void BackPropagate(VectorBatch outputgradient, VectorBatch input);
        public abstract void Update(AdaptationStrategy strategy);
            // void Run(NetworkVector input); defined in NetComponent
        #endregion

        #region protected fields
        protected WeightsMatrix _weightsGradientAccumulator;
        protected NetworkVector _biasesGradientAccumulator;
        #endregion

        #region public abstract properties
        public abstract WeightsMatrix Weights { get; }
        public abstract NetworkVector Biases { get; }
        #endregion

        #region public properties
        public NetworkVector VectorInput { get; protected set; }
        public VectorBatch BatchInput { get; protected set; }
        #endregion

        #region constructors
        public TrainableComponent(int numberofoutputs, int numberofinputs)
        {
            _weightsGradientAccumulator = new WeightsMatrix(numberofoutputs, numberofinputs);
            _biasesGradientAccumulator = new NetworkVector(numberofoutputs);
        }
        #endregion


        #region public abstract methods
        public abstract NetworkVector BiasesGradient(NetworkVector outputgradient);
        public abstract WeightsMatrix WeightsGradient(NetworkVector outputgradient);
        #endregion

        #region public methods
        public override NetworkVector Run(NetworkVector input)
        {
            VectorInput = input;
            BatchInput = null;
            return _run(input);
        }

        public override VectorBatch Run(VectorBatch inputbatch)
        {
            VectorInput = null;
            BatchInput = inputbatch;
            return _run(inputbatch);
        }
        #endregion

        #region protected abstract methods
        protected abstract NetworkVector _run(NetworkVector input);
        protected abstract VectorBatch _run(VectorBatch inputs);
        #endregion
    }
}
