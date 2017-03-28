using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public abstract class CostFunction
    {
        public abstract double Cost(NetworkVector target, NetworkVector vector);
        public abstract NetworkVector Gradient(NetworkVector target, NetworkVector vector);
    }


    public class SquaredError : CostFunction
    {
        public override double Cost(NetworkVector target, NetworkVector vector)
        {
            return 
                NetworkVector.ApplyFunctionComponentWise(
                    target, 
                    vector, 
                    (x, y) => (x - y) * (x - y)
                    ).SumValues() / 2;
        }

        public override NetworkVector Gradient(NetworkVector target, NetworkVector vector)
        {
            NetworkVector result = vector.Copy();
            result.Subtract(target);
            return result;
        }
    }

    public class CrossEntropy : CostFunction
    {
        public override double Cost(NetworkVector target, NetworkVector vector)
        {
            return
                NetworkVector.ApplyFunctionComponentWise(
                    target,
                    vector,
                    (x, y) => Math.Log(y) * x
                    ).SumValues() / vector.Dimension;
        }

        public override NetworkVector Gradient(NetworkVector target, NetworkVector vector)
        {
            return NetworkVector.ApplyFunctionComponentWise(
                target,
                vector,
                (x, y) => 1 / y
                );
        }
    }

    public class SoftMaxWithCrossEntropy : CostFunction
    {
        private NetworkVector _workingVector;
        private double _sum;

        public override double Cost(NetworkVector target, NetworkVector vector)
        {            
            _workingVector = NetworkVector.ApplyFunctionComponentWise(vector, x => Math.Exp(x));
            _sum = _workingVector.SumValues();
            _workingVector = NetworkVector.ApplyFunctionComponentWise(_workingVector, x => x / _sum);


            return
                NetworkVector.ApplyFunctionComponentWise(
                    target,
                    _workingVector,
                    (x, y) => Math.Log(y) * x
                    ).SumValues() / vector.Dimension;
        }

        public override NetworkVector Gradient(NetworkVector target, NetworkVector vector)
        {
            return NetworkVector.ApplyFunctionComponentWise(
                target,
                vector,
                (x, y) => x - y
                );
        }
    }
}
    