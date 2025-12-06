-- Create user_integrations table for third-party app connections
CREATE TABLE public.user_integrations (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  provider TEXT NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT true,
  synced_data JSONB DEFAULT '{}',
  last_sync TIMESTAMP WITH TIME ZONE,
  access_token TEXT,
  refresh_token TEXT,
  token_expires_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  UNIQUE(user_id, provider)
);

-- Enable Row Level Security
ALTER TABLE public.user_integrations ENABLE ROW LEVEL SECURITY;

-- Create policies for user access
CREATE POLICY "Users can view their own integrations" 
ON public.user_integrations 
FOR SELECT 
USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own integrations" 
ON public.user_integrations 
FOR INSERT 
WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own integrations" 
ON public.user_integrations 
FOR UPDATE 
USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own integrations" 
ON public.user_integrations 
FOR DELETE 
USING (auth.uid() = user_id);

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION public.update_user_integrations_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic timestamp updates
CREATE TRIGGER update_user_integrations_updated_at
  BEFORE UPDATE ON public.user_integrations
  FOR EACH ROW
  EXECUTE FUNCTION public.update_user_integrations_updated_at();