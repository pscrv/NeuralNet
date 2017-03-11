using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public abstract class NetworkComponent
    {
        public abstract int NumberOfInputs { get; }
        public abstract int NumberOfOutputs { get; }

        public abstract NetworkVector Output { get; }
        public abstract NetworkVector InputGradient { get; }        

        public abstract void Run(NetworkVector input);
        public abstract void BackPropagate(NetworkVector outputgradient);
    }
}
