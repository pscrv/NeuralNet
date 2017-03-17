using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public abstract class TrainableComponent : NetComponent
    {
        public abstract NetworkMatrix Weights { get; }
        public abstract NetworkVector Biases { get; }

        public abstract NetworkVector BiasesGradient(NetworkVector outputgradient);
        public abstract NetworkMatrix WeightsGradient(NetworkVector outputgradient);
        public abstract void Update(NetworkVector biasesdelta, NetworkMatrix weightsdelta);

        public StateGradient GetStateGradient(NetworkVector outputgradient)
        {
            return new StateGradient(
                this,
                WeightsGradient(outputgradient), 
                BiasesGradient(outputgradient));
        }
    }


    public class StateGradient
    {
        public TrainableComponent Component { get; private set; }
        public NetworkMatrix Weights { get; private set; }
        public NetworkVector Biases { get; private set; }

        public StateGradient(TrainableComponent component, NetworkMatrix weightsgradient, NetworkVector biasesgradient)
        {
            Component = component;
            Weights = weightsgradient;
            Biases = biasesgradient;
        }
    }
}
