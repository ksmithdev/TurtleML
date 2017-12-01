using System;
using System.Collections.Generic;
using System.Linq;

namespace TurtleML
{
    public class TrainingSet : List<Tuple<Tensor, Tensor>>
    {
        public TrainingSet(IEnumerable<Tuple<Tensor, Tensor>> collection) : base(collection)
        {
        }

        public TrainingSet() : base()
        {
        }

        public void Shuffle()
        {
            var random = new Random();

            int n = Count;
            while (n > 1)
            {
                int k = random.Next(n--);

                var value = this[n];
                this[n] = this[k];
                this[k] = value;
            }
        }

        public (TrainingSet, TrainingSet) Split(float percent)
        {
            var offset = (int)(Count * percent);

            return Split(offset);
        }

        public (TrainingSet, TrainingSet) Split(int index)
        {
            var set1 = new TrainingSet(this.Take(index));
            var set2 = new TrainingSet(this.Skip(index));

            return (set1, set2);
        }
    }
}