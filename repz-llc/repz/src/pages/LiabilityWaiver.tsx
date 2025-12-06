import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from "@/ui/atoms/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { ArrowLeft, AlertTriangle, Phone, Mail, Heart, Activity } from 'lucide-react';
import { RepzLogo } from '@/ui/organisms/RepzLogo';

export default function LiabilityWaiver() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-slate-900 to-black">
      {/* Header */}
      <div className="bg-white/95 backdrop-blur-sm shadow-lg border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate('/')}
            className="text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-all duration-200"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to REPZ
          </Button>
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-red-500 to-orange-500 rounded-lg flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-sm">R</span>
            </div>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-red-500 to-orange-500 font-bold text-2xl tracking-wider">REPZ</span>
          </div>
          <div className="text-center sm:text-right">
            <p className="text-sm text-slate-600 font-medium">Last updated</p>
            <p className="text-xs text-slate-500">February 4, 2025</p>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 lg:py-12">
        <Card className="shadow-2xl border-0 bg-white/95 backdrop-blur-sm">
          <CardHeader className="bg-gradient-to-r from-red-50 via-orange-50 to-red-50 border-b border-gray-200">
            <div className="text-center py-8">
              <CardTitle className="text-4xl md:text-5xl font-bold text-slate-900 mb-4 flex items-center justify-center gap-4">
                <div className="w-12 h-12 bg-gradient-to-br from-red-500 to-orange-500 rounded-xl flex items-center justify-center shadow-lg">
                  <AlertTriangle className="h-6 w-6 text-white" />
                </div>
                Liability Waiver & Release
              </CardTitle>
              <p className="text-slate-600 text-lg max-w-2xl mx-auto leading-relaxed">
                Exercise Risk Acknowledgment and Comprehensive Waiver of Claims for Fitness Activities
              </p>
              <div className="flex items-center justify-center gap-2 mt-4 text-sm text-slate-500">
                <span className="w-2 h-2 bg-red-500 rounded-full"></span>
                <span>Legal Protection Required</span>
              </div>
            </div>
          </CardHeader>
          
          <CardContent className="prose prose-slate max-w-none p-8">
            
            {/* Emergency Warning */}
            <div className="bg-red-100 border-2 border-red-300 rounded-lg p-6 mb-8">
              <div className="flex items-start gap-4">
                <Heart className="h-8 w-8 text-red-600 mt-1 flex-shrink-0" />
                <div>
                  <h2 className="text-2xl font-bold text-red-900 mb-3">EMERGENCY WARNING</h2>
                  <p className="text-red-800 font-semibold mb-3">
                    STOP EXERCISING IMMEDIATELY if you experience:
                  </p>
                  <ul className="list-disc pl-6 space-y-1 text-red-800 font-medium">
                    <li>Chest pain, pressure, or tightness</li>
                    <li>Severe shortness of breath</li>
                    <li>Dizziness, lightheadedness, or fainting</li>
                    <li>Unusual fatigue or weakness</li>
                    <li>Nausea or vomiting during exercise</li>
                    <li>Irregular heartbeat or heart palpitations</li>
                    <li>Pain in arms, jaw, neck, or back</li>
                  </ul>
                  <p className="text-red-800 font-bold mt-4 text-lg">
                    CALL 911 IMMEDIATELY IF YOU EXPERIENCE ANY OF THESE SYMPTOMS
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-8">
              
              {/* Acknowledgment of Risks */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">1. Acknowledgment of Exercise Risks</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  I understand and acknowledge that participation in any exercise program, including the fitness coaching services provided by REPZ, involves inherent risks that cannot be eliminated regardless of the care taken to avoid injuries.
                </p>
                
                <h3 className="text-xl font-semibold text-slate-800 mb-3">Specific Risks Include But Are Not Limited To:</h3>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li><strong>Cardiovascular complications:</strong> Heart attack, stroke, irregular heartbeat, high blood pressure</li>
                  <li><strong>Musculoskeletal injuries:</strong> Muscle strains, sprains, tears, joint injuries, fractures</li>
                  <li><strong>Respiratory issues:</strong> Shortness of breath, asthma attacks, oxygen deprivation</li>
                  <li><strong>Metabolic complications:</strong> Dehydration, heat exhaustion, hypoglycemia</li>
                  <li><strong>Neurological effects:</strong> Dizziness, fainting, loss of consciousness</li>
                  <li><strong>Equipment-related injuries:</strong> Falls, equipment malfunction, improper use</li>
                  <li><strong>Overexertion:</strong> Fatigue, overtraining syndrome, burnout</li>
                  <li><strong>Death:</strong> In rare cases, exercise may result in sudden cardiac death or other fatal complications</li>
                </ul>

                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
                  <div className="flex items-start gap-3">
                    <Activity className="h-5 w-5 text-yellow-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-yellow-900">Individual Risk Factors</h4>
                      <p className="text-yellow-800 text-sm mt-1">
                        Your individual risk may be higher if you have pre-existing medical conditions, are sedentary, or have not received medical clearance for exercise.
                      </p>
                    </div>
                  </div>
                </div>
              </section>

              {/* Medical Clearance */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">2. Medical Clearance and Health Disclosure</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  I acknowledge that:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>I have been advised to obtain medical clearance from a qualified healthcare provider before beginning any exercise program</li>
                  <li>I am responsible for consulting with my physician regarding my ability to participate in physical exercise</li>
                  <li>I must disclose any medical conditions, medications, or physical limitations that may affect my ability to exercise safely</li>
                  <li>I understand that REPZ coaches are not medical professionals and cannot provide medical advice</li>
                  <li>I will immediately inform my coach of any changes in my health status</li>
                </ul>
                
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-5 w-5 text-red-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-red-900">Medical Disclaimer</h4>
                      <p className="text-red-800 text-sm mt-1">
                        REPZ services are not intended to diagnose, treat, cure, or prevent any disease. Always consult your healthcare provider for medical advice.
                      </p>
                    </div>
                  </div>
                </div>
              </section>

              {/* Assumption of Risk */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">3. Voluntary Assumption of Risk</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  I voluntarily assume all risks associated with my participation in REPZ fitness programs, including but not limited to the risks described above. I understand that:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>My participation is entirely voluntary and at my own risk</li>
                  <li>I am free to stop participating at any time</li>
                  <li>I am responsible for exercising within my physical capabilities and limitations</li>
                  <li>I will not hold REPZ responsible for any injuries or health complications that may result</li>
                  <li>I understand that individual results may vary significantly</li>
                </ul>
              </section>

              {/* Release and Waiver */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">4. Release and Waiver of Claims</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  In consideration for being allowed to participate in REPZ fitness programs, I hereby:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li><strong>RELEASE</strong> REPZ, its owners, employees, contractors, and affiliates from any and all liability, claims, demands, actions, or causes of action</li>
                  <li><strong>WAIVE</strong> any right to sue or claim compensation for injuries, damages, or losses arising from my participation</li>
                  <li><strong>HOLD HARMLESS</strong> REPZ from any liability for injuries to third parties that may result from my participation</li>
                  <li><strong>INDEMNIFY</strong> REPZ against any claims brought by others as a result of my participation</li>
                </ul>
                
                <p className="text-slate-700 leading-relaxed mb-4">
                  This release covers all claims whether based on negligence, breach of contract, or any other legal theory, and includes claims for:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700">
                  <li>Personal injury, illness, or death</li>
                  <li>Property damage or loss</li>
                  <li>Economic losses or damages</li>
                  <li>Pain and suffering</li>
                  <li>Medical expenses and rehabilitation costs</li>
                  <li>Lost wages or earning capacity</li>
                </ul>
              </section>

              {/* Emergency Authorization */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">5. Emergency Medical Authorization</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  In the event of a medical emergency during my participation in REPZ programs, I authorize:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>REPZ staff to call emergency medical services (911) on my behalf</li>
                  <li>Emergency medical personnel to provide necessary medical treatment</li>
                  <li>Transportation to the nearest appropriate medical facility</li>
                  <li>REPZ to contact my emergency contacts</li>
                </ul>
                <p className="text-slate-700 leading-relaxed">
                  I understand that I am financially responsible for any emergency medical care or transportation costs.
                </p>
              </section>

              {/* Photography and Media */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">6. Photography and Media Release</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  I grant REPZ permission to use my name, likeness, voice, and biographical information in photographs, videos, and other media for:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>Marketing and promotional materials</li>
                  <li>Educational content and testimonials</li>
                  <li>Social media and website content</li>
                  <li>Program documentation and improvement</li>
                </ul>
                <p className="text-slate-700 leading-relaxed">
                  I waive any right to compensation for such use and understand that this permission is perpetual unless revoked in writing.
                </p>
              </section>

              {/* Legal Provisions */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">7. Legal Provisions</h2>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li><strong>Severability:</strong> If any provision of this waiver is found invalid, the remaining provisions shall remain in full force</li>
                  <li><strong>Governing Law:</strong> This waiver shall be governed by the laws of [Your Jurisdiction]</li>
                  <li><strong>Binding Effect:</strong> This waiver binds my heirs, executors, administrators, and assigns</li>
                  <li><strong>Integration:</strong> This waiver represents the complete agreement regarding liability and risk assumption</li>
                  <li><strong>Voluntary Execution:</strong> I have read this waiver carefully and sign it voluntarily with full understanding of its contents</li>
                </ul>
              </section>

              {/* Acknowledgment */}
              <section className="bg-slate-50 border border-slate-200 rounded-lg p-6">
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">Acknowledgment and Agreement</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  By participating in REPZ fitness programs, I acknowledge that:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>I have read this entire waiver and understand its contents</li>
                  <li>I have had the opportunity to ask questions about any provisions I do not understand</li>
                  <li>I am signing this waiver voluntarily without coercion</li>
                  <li>I am of legal age and mentally competent to enter into this agreement</li>
                  <li>No oral or written representations have been made to me that are inconsistent with this waiver</li>
                </ul>
                
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 mt-4">
                  <p className="text-red-800 font-semibold text-center">
                    THIS IS A LEGAL DOCUMENT THAT AFFECTS YOUR RIGHTS.<br />
                    READ CAREFULLY BEFORE AGREEING TO PARTICIPATE.
                  </p>
                </div>
              </section>

              {/* Emergency Contact Information */}
              <section className="bg-red-50 border border-red-200 rounded-lg p-6">
                <h2 className="text-2xl font-semibold text-red-900 mb-4">Emergency Contact Information</h2>
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-red-800">
                    <Phone className="h-5 w-5" />
                    <span className="font-semibold">Emergency Services: 911</span>
                  </div>
                  <div className="flex items-center gap-2 text-red-800">
                    <Phone className="h-5 w-5" />
                    <span>REPZ Emergency Line: +1 (415) 992-9792</span>
                  </div>
                  <div className="flex items-center gap-2 text-red-800">
                    <Mail className="h-5 w-5" />
                    <span>Emergency Email: emergency@repz.com</span>
                  </div>
                </div>
              </section>

            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}