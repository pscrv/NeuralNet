using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.NetComponent
{
    public abstract class NetComponent
    {
        public abstract int NumberOfInputs { get; }
        public abstract int NumberOfOutputs { get; }
        public abstract NetworkVector Output { get; protected set; }
        public abstract NetworkVector InputGradient(NetworkVector outputgradient);        
        public abstract void Run(NetworkVector input);
    }    
}
