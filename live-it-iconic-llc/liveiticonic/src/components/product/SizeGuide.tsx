import React from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Ruler } from 'lucide-react';

interface SizeGuideProps {
  category: string;
}

/**
 * SizeGuide component displays category-specific size charts in a modal dialog
 *
 * Renders a modal with size measurement guides that vary by product category (tees, hoodies, etc.).
 * Includes measurement instructions and provides reference tables with chest, length, and sleeve
 * dimensions. Triggered by a button with ruler icon.
 *
 * @component
 * @param {SizeGuideProps} props - Component props
 * @param {string} props.category - Product category to determine which size chart to display (e.g., 'tee', 'hoodie')
 *
 * @example
 * <SizeGuide category="t-shirt" />
 *
 * @remarks
 * - Chart data is hardcoded for 'tee'/'shirt' and 'hoodie'/'sweatshirt' categories
 * - Unknown categories default to a generic 'One Size Fits Most' chart
 * - Measurements are in inches
 * - Dialog is scrollable for small screens
 */
export const SizeGuide: React.FC<SizeGuideProps> = ({ category }) => {
  // Size chart data based on category
  const getSizeChart = () => {
    if (category.toLowerCase().includes('tee') || category.toLowerCase().includes('shirt')) {
      return {
        headers: ['Size', 'Chest (inches)', 'Length (inches)', 'Sleeve (inches)'],
        rows: [
          ['XS', '34-36', '26', '32'],
          ['S', '36-38', '27', '33'],
          ['M', '40-42', '28', '34'],
          ['L', '44-46', '29', '35'],
          ['XL', '48-50', '30', '36'],
          ['XXL', '52-54', '31', '37'],
        ],
      };
    } else if (
      category.toLowerCase().includes('hoodie') ||
      category.toLowerCase().includes('sweatshirt')
    ) {
      return {
        headers: ['Size', 'Chest (inches)', 'Length (inches)', 'Sleeve (inches)'],
        rows: [
          ['S', '38-40', '28', '33'],
          ['M', '42-44', '29', '34'],
          ['L', '46-48', '30', '35'],
          ['XL', '50-52', '31', '36'],
          ['XXL', '54-56', '32', '37'],
        ],
      };
    } else {
      return {
        headers: ['Size', 'One Size'],
        rows: [['One Size', 'Fits Most']],
      };
    }
  };

  const sizeChart = getSizeChart();

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <Ruler className="w-4 h-4 mr-2" />
          Size Guide
        </Button>
      </DialogTrigger>
      <DialogContent className="bg-lii-ink border-lii-gold/10 max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-lii-cloud font-display">Size Guide</DialogTitle>
          <DialogDescription className="text-lii-ash">Measurements and tips to choose the right size.</DialogDescription>
        </DialogHeader>
        <div className="mt-4">
          <Table>
            <TableHeader>
              <TableRow className="border-lii-gold/10">
                {sizeChart.headers.map(header => (
                  <TableHead key={header} className="text-lii-cloud">
                    {header}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {sizeChart.rows.map((row, index) => (
                <TableRow key={index} className="border-lii-gold/10">
                  {row.map((cell, cellIndex) => (
                    <TableCell key={cellIndex} className="text-lii-ash">
                      {cell}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <div className="mt-6 p-4 bg-lii-gold/5 rounded-lg border border-lii-gold/20">
            <p className="text-lii-ash font-ui text-sm">
              <strong className="text-lii-cloud">Measurement Tips:</strong> Measure your chest at
              the fullest point, length from shoulder to hem, and sleeve from shoulder to wrist.
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default SizeGuide;
