using MathNet.Numerics.LinearAlgebra;
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
        public abstract double Cost(VectorBatch target, VectorBatch batch);
        public abstract NetworkVector Gradient(NetworkVector target, NetworkVector vector);
        public abstract VectorBatch Gradient(VectorBatch target, VectorBatch batch);
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

        public override double Cost(VectorBatch target, VectorBatch batch)
        {
            throw new NotImplementedException();
        }

        public override NetworkVector Gradient(NetworkVector target, NetworkVector vector)
        {
            NetworkVector result = vector.Copy();
            result.Subtract(target);
            return result;
        }

        public override VectorBatch Gradient(VectorBatch target, VectorBatch batch)
        {
            throw new NotImplementedException();
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

        public override double Cost(VectorBatch target, VectorBatch batch)
        {
            return
                target.AsMatrix().Map2((x, y) => Math.Log(y) * x, batch.AsMatrix()).RowSums().Sum() / (batch.Count * batch.Dimension);
        }

        public override NetworkVector Gradient(NetworkVector target, NetworkVector vector)
        {
            return NetworkVector.ApplyFunctionComponentWise(
                target,
                vector,
                (x, y) => 1 / y
                );
        }

        public override VectorBatch Gradient(VectorBatch target, VectorBatch batch)
        {
            return new VectorBatch(
                target.AsMatrix().Map2((x, y) => 1 / y, batch.AsMatrix())
                );
        }
    }

    public class SoftMaxWithCrossEntropy : CostFunction
    {
        private NetworkVector _workingVector;
        private Matrix<double> _workingBatchMatrix;

        public override double Cost(NetworkVector target, NetworkVector vector)
        {            
            _workingVector = NetworkVector.ApplyFunctionComponentWise(vector, x => Math.Exp(x));
            _workingBatchMatrix = null;
            double sum = _workingVector.SumValues();
            _workingVector = NetworkVector.ApplyFunctionComponentWise(_workingVector, x => x / sum);


            return
                NetworkVector.ApplyFunctionComponentWise(
                    target,
                    _workingVector,
                    (x, y) => Math.Log(y) * x
                    ).SumValues() / vector.Dimension;
        }

        public override double Cost(VectorBatch target, VectorBatch batch)
        {
            _workingVector = null;
            _workingBatchMatrix = batch.AsMatrix().Map( x => Math.Exp(x));
            Vector<double> sum = _workingBatchMatrix.RowSums();
            _workingBatchMatrix =  _workingBatchMatrix.NormalizeRows(1.0) ;
            _workingBatchMatrix = target.AsMatrix().Map2((x, y) => Math.Log(y) * x, _workingBatchMatrix);

            return _workingBatchMatrix.RowSums().Sum() / (batch.Count * batch.Dimension);
        }

        public override NetworkVector Gradient(NetworkVector target, NetworkVector vector)
        {
            return NetworkVector.ApplyFunctionComponentWise(
                target,
                vector,
                (x, y) => x - y
                );
        }

        public override VectorBatch Gradient(VectorBatch target, VectorBatch batch)
        {
            return new VectorBatch(target.AsMatrix() - batch.AsMatrix());
        }
    }
}
    