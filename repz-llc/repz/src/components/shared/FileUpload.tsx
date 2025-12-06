import React, { useState, useCallback, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import { supabase } from '@/integrations/supabase/client';
import { 
  Upload, 
  X, 
  Camera, 
  FileText, 
  Image, 
  Download,
  Eye,
  Trash2,
  CheckCircle,
  AlertCircle,
  File
} from 'lucide-react';

interface FileUploadProps {
  bucket: string;
  folder?: string;
  allowedTypes?: string[];
  maxFileSize?: number; // in MB
  maxFiles?: number;
  onUploadComplete?: (files: UploadedFile[]) => void;
  existingFiles?: UploadedFile[];
  title?: string;
  description?: string;
  uploadType?: 'progress-photos' | 'documents' | 'bloodwork' | 'general';
}

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  url: string;
  uploadedAt: string;
  category?: string;
}

interface UploadProgress {
  file: File;
  progress: number;
  status: 'uploading' | 'completed' | 'error';
  error?: string;
}

export const FileUpload: React.FC<FileUploadProps> = ({
  bucket,
  folder = '',
  allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'application/pdf'],
  maxFileSize = 10,
  maxFiles = 5,
  onUploadComplete,
  existingFiles = [],
  title = "File Upload",
  description = "Upload your files",
  uploadType = 'general'
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress[]>([]);
  const [files, setFiles] = useState<UploadedFile[]>(existingFiles);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    handleFiles(droppedFiles);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      handleFiles(selectedFiles);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const validateFile = (file: File): string | null => {
    if (!allowedTypes.includes(file.type)) {
      return `File type ${file.type} is not allowed`;
    }
    
    if (file.size > maxFileSize * 1024 * 1024) {
      return `File size exceeds ${maxFileSize}MB limit`;
    }
    
    return null;
  };

  const handleFiles = async (selectedFiles: File[]) => {
    if (files.length + selectedFiles.length > maxFiles) {
      toast({
        title: "Too many files",
        description: `Maximum ${maxFiles} files allowed`,
        variant: "destructive"
      });
      return;
    }

    const validFiles: File[] = [];
    
    for (const file of selectedFiles) {
      const error = validateFile(file);
      if (error) {
        toast({
          title: "Invalid file",
          description: `${file.name}: ${error}`,
          variant: "destructive"
        });
      } else {
        validFiles.push(file);
      }
    }

    if (validFiles.length === 0) return;

    // Initialize upload progress
    const newProgress = validFiles.map(file => ({
      file,
      progress: 0,
      status: 'uploading' as const
    }));
    
    setUploadProgress(prev => [...prev, ...newProgress]);

    // Upload files
    for (let i = 0; i < validFiles.length; i++) {
      await uploadFile(validFiles[i], i);
    }
  };

  const uploadFile = async (file: File, index: number) => {
    try {
      const fileExt = file.name.split('.').pop();
      const fileName = `${Date.now()}-${Math.random().toString(36).substring(2)}.${fileExt}`;
      const filePath = folder ? `${folder}/${fileName}` : fileName;

      // Update progress to show starting
      setUploadProgress(prev =>
        prev.map((p, i) =>
          i === index ? { ...p, progress: 10 } : p
        )
      );

      const { error: uploadError } = await supabase.storage
        .from(bucket)
        .upload(filePath, file, {
          cacheControl: '3600',
          upsert: false
        });

      if (uploadError) throw uploadError;

      // Update progress to show completion
      setUploadProgress(prev =>
        prev.map((p, i) =>
          i === index ? { ...p, progress: 100, status: 'completed' } : p
        )
      );

      // Get public URL
      const { data: { publicUrl } } = supabase.storage
        .from(bucket)
        .getPublicUrl(filePath);

      const newFile: UploadedFile = {
        id: fileName,
        name: file.name,
        size: file.size,
        type: file.type,
        url: publicUrl,
        uploadedAt: new Date().toISOString(),
        category: uploadType
      };

      setFiles(prev => [...prev, newFile]);

      toast({
        title: "Upload successful",
        description: `${file.name} has been uploaded`,
      });

    } catch (error) {
      console.error('Upload error:', error);
      
      setUploadProgress(prev =>
        prev.map((p, i) =>
          i === index ? { 
            ...p, 
            status: 'error', 
            error: error instanceof Error ? error.message : 'Upload failed'
          } : p
        )
      );

      toast({
        title: "Upload failed",
        description: `Failed to upload ${file.name}`,
        variant: "destructive"
      });
    }
  };

  const removeFile = async (fileToRemove: UploadedFile) => {
    try {
      // Remove from storage
      const filePath = folder ? `${folder}/${fileToRemove.id}` : fileToRemove.id;
      const { error } = await supabase.storage
        .from(bucket)
        .remove([filePath]);

      if (error) throw error;

      // Remove from state
      setFiles(prev => prev.filter(f => f.id !== fileToRemove.id));
      
      toast({
        title: "File deleted",
        description: `${fileToRemove.name} has been removed`,
      });

    } catch (error) {
      console.error('Delete error:', error);
      toast({
        title: "Delete failed",
        description: "Failed to delete file",
        variant: "destructive"
      });
    }
  };

  const clearUploadProgress = () => {
    setUploadProgress([]);
  };

  const getFileIcon = (type: string) => {
    if (type.startsWith('image/')) return <Image className="h-4 w-4" />;
    if (type === 'application/pdf') return <FileText className="h-4 w-4" />;
    return <File className="h-4 w-4" />;
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            {title}
          </CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-4">
          {/* Upload Area */}
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              isDragOver 
                ? 'border-primary bg-primary/10' 
                : 'border-muted-foreground/25 hover:border-primary/50'
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="space-y-4">
              {uploadType === 'progress-photos' ? (
                <Camera className="h-12 w-12 mx-auto text-muted-foreground" />
              ) : (
                <Upload className="h-12 w-12 mx-auto text-muted-foreground" />
              )}
              
              <div>
                <p className="text-lg font-medium">
                  {isDragOver ? 'Drop files here' : 'Drag & drop files here'}
                </p>
                <p className="text-sm text-muted-foreground mt-1">
                  or click to browse files
                </p>
              </div>
              
              <Button
                onClick={() => fileInputRef.current?.click()}
                variant="outline"
                disabled={files.length >= maxFiles}
              >
                Browse Files
              </Button>
              
              <div className="text-xs text-muted-foreground space-y-1">
                <p>Allowed: {allowedTypes.map(type => type.split('/')[1]).join(', ')}</p>
                <p>Max size: {maxFileSize}MB per file</p>
                <p>Max files: {maxFiles}</p>
              </div>
            </div>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept={allowedTypes.join(',')}
            onChange={handleFileInput}
            className="hidden"
          />

          {/* Upload Progress */}
          {uploadProgress.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">Upload Progress</CardTitle>
                  <Button onClick={clearUploadProgress} variant="ghost" size="sm">
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                {uploadProgress.map((progress, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="truncate">{progress.file.name}</span>
                      <div className="flex items-center gap-2">
                        {progress.status === 'completed' && (
                          <CheckCircle className="h-4 w-4 text-green-500" />
                        )}
                        {progress.status === 'error' && (
                          <AlertCircle className="h-4 w-4 text-red-500" />
                        )}
                        <span>{progress.progress}%</span>
                      </div>
                    </div>
                    <Progress value={progress.progress} className="h-2" />
                    {progress.error && (
                      <p className="text-xs text-red-500">{progress.error}</p>
                    )}
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

          {/* Uploaded Files */}
          {files.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  Uploaded Files ({files.length})
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-3">
                  {files.map((file) => (
                    <div 
                      key={file.id}
                      className="flex items-center justify-between p-3 border rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        {getFileIcon(file.type)}
                        <div>
                          <p className="text-sm font-medium">{file.name}</p>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <span>{formatFileSize(file.size)}</span>
                            <span>•</span>
                            <span>{new Date(file.uploadedAt).toLocaleDateString()}</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-1">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => window.open(file.url, '_blank')}
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                        
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => {
                            const link = document.createElement('a');
                            link.href = file.url;
                            link.download = file.name;
                            link.click();
                          }}
                        >
                          <Download className="h-4 w-4" />
                        </Button>
                        
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => removeFile(file)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* File Limits Info */}
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>{files.length} of {maxFiles} files</span>
            {files.length >= maxFiles && (
              <Badge variant="secondary">Maximum files reached</Badge>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};