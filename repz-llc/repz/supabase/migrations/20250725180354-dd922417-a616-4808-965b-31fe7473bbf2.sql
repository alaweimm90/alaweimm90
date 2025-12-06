-- Add foreign key constraints to messages table for proper joins
ALTER TABLE public.messages 
ADD CONSTRAINT messages_sender_id_fkey 
FOREIGN KEY (sender_id) REFERENCES auth.users(id) ON DELETE CASCADE;

ALTER TABLE public.messages 
ADD CONSTRAINT messages_recipient_id_fkey 
FOREIGN KEY (recipient_id) REFERENCES auth.users(id) ON DELETE CASCADE;