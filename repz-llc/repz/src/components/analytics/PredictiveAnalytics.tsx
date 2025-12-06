import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { 
  TrendingUp, 
  Brain, 
  Target, 
  AlertTriangle,
  Users,
  Calendar,
  Activity,
  Star,
  Download,
  RefreshCw
} from 'lucide-react';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { supabase } from '@/integrations/supabase/client';

interface PredictionModel {
  name: string;
  accuracy: number;
  lastTrained: string;
  status: 'active' | 'training' | 'needs_update';
  predictions: number;
}

interface ClientRiskScore {
  clientId: string;
  clientName: string;
  riskScore: number;
  confidence: number;
  riskFactors: { factor: string; weight: number }[];
  recommendation: string;
  tier: string;
}

interface SuccessPrediction {
  clientId: string;
  clientName: string;
  successProbability: number;
  timeToGoal: number;
  keyFactors: string[];
  recommendedActions: string[];
}

interface RevenueForecasting {
  period: string;
  predictedRevenue: number;
  confidence: number;
  factors: { factor: string; impact: number }[];
}

const PredictiveAnalytics: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [models, setModels] = useState<PredictionModel[]>([]);
  const [churnRisks, setChurnRisks] = useState<ClientRiskScore[]>([]);
  const [successPredictions, setSuccessPredictions] = useState<SuccessPrediction[]>([]);
  const [revenueForecasts, setRevenueForecasts] = useState<RevenueForecasting[]>([]);

  useEffect(() => {
    fetchPredictiveData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchPredictiveData = async () => {
    setIsLoading(true);
    try {
      // Mock data - in production, this would call AI/ML services
      await Promise.all([
        fetchModelStatus(),
        fetchChurnRiskScores(),
        fetchSuccessPredictions(),
        fetchRevenueForecasts()
      ]);
    } catch (error) {
      console.error('Error fetching predictive analytics:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchModelStatus = async () => {
    const mockModels: PredictionModel[] = [
      {
        name: 'Churn Prediction Model',
        accuracy: 87.5,
        lastTrained: '2024-01-15',
        status: 'active',
        predictions: 143
      },
      {
        name: 'Success Prediction Model', 
        accuracy: 82.3,
        lastTrained: '2024-01-12',
        status: 'active',
        predictions: 98
      },
      {
        name: 'Revenue Forecasting Model',
        accuracy: 91.2,
        lastTrained: '2024-01-18',
        status: 'active',
        predictions: 24
      },
      {
        name: 'Engagement Prediction Model',
        accuracy: 76.8,
        lastTrained: '2024-01-05',
        status: 'needs_update',
        predictions: 67
      }
    ];
    setModels(mockModels);
  };

  const fetchChurnRiskScores = async () => {
    const mockRisks: ClientRiskScore[] = [
      {
        clientId: 'client-001',
        clientName: 'John Smith',
        riskScore: 85,
        confidence: 92,
        riskFactors: [
          { factor: 'Missed last 3 check-ins', weight: 0.35 },
          { factor: 'Declining workout frequency', weight: 0.28 },
          { factor: 'No coach interaction in 2 weeks', weight: 0.22 },
          { factor: 'Below average goal progress', weight: 0.15 }
        ],
        recommendation: 'Immediate coach outreach recommended. Schedule personal check-in call.',
        tier: 'Adaptive'
      },
      {
        clientId: 'client-002', 
        clientName: 'Sarah Johnson',
        riskScore: 72,
        confidence: 87,
        riskFactors: [
          { factor: 'Payment method expired', weight: 0.40 },
          { factor: 'Reduced app usage', weight: 0.30 },
          { factor: 'Low satisfaction scores', weight: 0.30 }
        ],
        recommendation: 'Update payment info and offer satisfaction survey.',
        tier: 'Performance'
      },
      {
        clientId: 'client-003',
        clientName: 'Mike Davis',
        riskScore: 68,
        confidence: 79,
        riskFactors: [
          { factor: 'Plateau in progress metrics', weight: 0.45 },
          { factor: 'Irregular workout schedule', weight: 0.35 },
          { factor: 'Limited goal adjustment', weight: 0.20 }
        ],
        recommendation: 'Suggest program modification and new goal setting session.',
        tier: 'Core'
      }
    ];
    setChurnRisks(mockRisks);
  };

  const fetchSuccessPredictions = async () => {
    const mockPredictions: SuccessPrediction[] = [
      {
        clientId: 'client-004',
        clientName: 'Emma Wilson',
        successProbability: 94,
        timeToGoal: 8,
        keyFactors: ['Consistent workout schedule', 'Regular coach interaction', 'Strong goal alignment'],
        recommendedActions: ['Continue current program', 'Consider goal advancement', 'Share success story']
      },
      {
        clientId: 'client-005',
        clientName: 'David Brown',
        successProbability: 78,
        timeToGoal: 12,
        keyFactors: ['Good nutrition adherence', 'Moderate exercise consistency'],
        recommendedActions: ['Increase workout frequency', 'Add accountability partner']
      }
    ];
    setSuccessPredictions(mockPredictions);
  };

  const fetchRevenueForecasts = async () => {
    const mockForecasts: RevenueForecasting[] = [
      {
        period: 'Next Month',
        predictedRevenue: 14200,
        confidence: 89,
        factors: [
          { factor: 'New subscriptions', impact: 0.35 },
          { factor: 'Retention rate', impact: 0.40 },
          { factor: 'Tier upgrades', impact: 0.25 }
        ]
      },
      {
        period: 'Next Quarter',
        predictedRevenue: 45600,
        confidence: 82,
        factors: [
          { factor: 'Seasonal trends', impact: 0.30 },
          { factor: 'Marketing campaigns', impact: 0.25 },
          { factor: 'Competition', impact: 0.20 },
          { factor: 'Product improvements', impact: 0.25 }
        ]
      }
    ];
    setRevenueForecasts(mockForecasts);
  };

  const retrainModel = async (modelName: string) => {
    setIsLoading(true);
    // Simulate model retraining
    setTimeout(() => {
      setModels(prev => prev.map(model => 
        model.name === modelName 
          ? { ...model, status: 'training' as const, lastTrained: new Date().toISOString().split('T')[0] }
          : model
      ));
      setIsLoading(false);
    }, 2000);
  };

  const exportPredictions = () => {
    // Export predictions as CSV or JSON
    console.log('Exporting prediction data...');
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'active':
        return <Badge className="bg-green-100 text-green-700">Active</Badge>;
      case 'training':
        return <Badge className="bg-blue-100 text-blue-700">Training</Badge>;
      case 'needs_update':
        return <Badge className="bg-red-100 text-red-700">Needs Update</Badge>;
      default:
        return <Badge variant="secondary">Unknown</Badge>;
    }
  };

  const getRiskColor = (score: number) => {
    if (score >= 80) return 'text-red-600 bg-red-50 border-red-200';
    if (score >= 60) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-green-600 bg-green-50 border-green-200';
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Brain className="h-6 w-6 text-purple-600" />
            Predictive Analytics
          </h2>
          <p className="text-muted-foreground">AI-powered insights and predictions</p>
        </div>
        <div className="flex items-center gap-3">
          <Button 
            variant="outline" 
            onClick={fetchPredictiveData}
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button onClick={exportPredictions}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Model Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {models.map((model, index) => (
          <Card key={index}>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm">{model.name}</CardTitle>
                {getStatusBadge(model.status)}
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Accuracy</span>
                  <span className="font-medium">{model.accuracy}%</span>
                </div>
                <Progress value={model.accuracy} />
                <div className="text-xs text-muted-foreground">
                  Last trained: {model.lastTrained}
                </div>
                <div className="text-xs text-muted-foreground">
                  {model.predictions} predictions made
                </div>
                {model.status === 'needs_update' && (
                  <Button 
                    size="sm" 
                    className="w-full mt-2"
                    onClick={() => retrainModel(model.name)}
                  >
                    Retrain Model
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Churn Risk Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red-500" />
              High Churn Risk Clients
            </CardTitle>
            <CardDescription>
              Clients with elevated risk of canceling their subscription
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {churnRisks.map((risk, index) => (
              <div key={index} className={`p-4 rounded-lg border ${getRiskColor(risk.riskScore)}`}>
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <p className="font-medium">{risk.clientName}</p>
                    <p className="text-sm text-muted-foreground">{risk.tier} tier</p>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-bold">{risk.riskScore}%</p>
                    <p className="text-xs text-muted-foreground">{risk.confidence}% confidence</p>
                  </div>
                </div>
                
                <div className="space-y-2 text-sm">
                  <p className="font-medium">Risk Factors:</p>
                  {risk.riskFactors.map((factor, factorIndex) => (
                    <div key={factorIndex} className="flex justify-between">
                      <span>{factor.factor}</span>
                      <span className="font-medium">{Math.round(factor.weight * 100)}%</span>
                    </div>
                  ))}
                </div>
                
                <Alert className="mt-3">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription className="text-sm">
                    <strong>Recommendation:</strong> {risk.recommendation}
                  </AlertDescription>
                </Alert>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Success Predictions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5 text-green-500" />
              Success Predictions
            </CardTitle>
            <CardDescription>
              Clients likely to achieve their goals successfully
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {successPredictions.map((prediction, index) => (
              <div key={index} className="p-4 rounded-lg border border-green-200 bg-green-50">
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <p className="font-medium">{prediction.clientName}</p>
                    <p className="text-sm text-muted-foreground">
                      Goal achievement in ~{prediction.timeToGoal} weeks
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-bold text-green-600">{prediction.successProbability}%</p>
                    <p className="text-xs text-muted-foreground">success probability</p>
                  </div>
                </div>
                
                <div className="space-y-2 text-sm">
                  <p className="font-medium">Key Success Factors:</p>
                  <ul className="list-disc list-inside space-y-1">
                    {prediction.keyFactors.map((factor, factorIndex) => (
                      <li key={factorIndex}>{factor}</li>
                    ))}
                  </ul>
                </div>
                
                <div className="mt-3 space-y-1">
                  <p className="text-sm font-medium">Recommended Actions:</p>
                  {prediction.recommendedActions.map((action, actionIndex) => (
                    <div key={actionIndex} className="text-xs bg-white p-2 rounded border">
                      {action}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Revenue Forecasting */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-blue-500" />
            Revenue Forecasting
          </CardTitle>
          <CardDescription>
            AI-powered revenue predictions based on current trends
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {revenueForecasts.map((forecast, index) => (
              <div key={index} className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold">{forecast.period}</h3>
                  <Badge className="bg-blue-100 text-blue-700">
                    {forecast.confidence}% confidence
                  </Badge>
                </div>
                
                <div className="text-center mb-4">
                  <p className="text-3xl font-bold text-blue-600">
                    {formatCurrency(forecast.predictedRevenue)}
                  </p>
                  <p className="text-sm text-muted-foreground">Predicted Revenue</p>
                </div>
                
                <div className="space-y-2">
                  <p className="text-sm font-medium">Contributing Factors:</p>
                  {forecast.factors.map((factor, factorIndex) => (
                    <div key={factorIndex} className="flex justify-between text-sm">
                      <span>{factor.factor}</span>
                      <span className="font-medium">{Math.round(factor.impact * 100)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PredictiveAnalytics;