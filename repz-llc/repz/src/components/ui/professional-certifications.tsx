import { Trophy, Award, Heart, FileText, Crown, Activity, Zap } from 'lucide-react';

export function ProfessionalCertifications() {
  const certifications = [
    { name: "NASM-CPT", icon: Trophy, color: "bg-gradient-to-br from-amber-700 to-yellow-800 border-amber-600" },
    { name: "CES", icon: Award, color: "bg-gradient-to-br from-blue-700 to-blue-800 border-blue-600" },
    { name: "Biolayne", icon: Heart, color: "bg-gradient-to-br from-red-800 to-red-900 border-red-700" },
    { name: "ISSA-SN", icon: FileText, color: "bg-gradient-to-br from-teal-700 to-teal-800 border-teal-600" },
    { name: "PhD", icon: Crown, color: "bg-gradient-to-br from-purple-800 to-purple-900 border-purple-700" },
    { name: "PED", icon: Activity, color: "bg-gradient-to-br from-green-700 to-green-800 border-green-600" },
    { name: "Author", icon: FileText, color: "bg-gradient-to-br from-amber-800 to-amber-900 border-amber-700" },
    { name: "AED", icon: Zap, color: "bg-gradient-to-br from-purple-700 to-red-800 border-purple-600" }
  ];

  return (
    <div className="mt-8">
      <h4 className="text-xl font-bold text-orange-200 mb-6">Professional Certifications</h4>
      <div className="grid grid-cols-4 gap-3">
        {certifications.map((cert) => {
          const IconComponent = cert.icon;
          return (
            <div key={cert.name} 
                 className={`${cert.color} backdrop-blur-sm border rounded-xl p-4 text-center hover:scale-105 transition-all duration-300 shadow-lg hover:shadow-orange-500/20 group`}>
              <IconComponent className="w-6 h-6 mx-auto mb-2 text-white group-hover:scale-110 transition-transform" />
              <div className="text-white font-semibold text-xs">{cert.name}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}