import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from "@/ui/atoms/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { ArrowLeft, AlertTriangle, Phone, Mail, Heart, Stethoscope, Pill } from 'lucide-react';
import { RepzLogo } from '@/ui/organisms/RepzLogo';

export default function HealthDisclaimer() {
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
            <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-blue-500 rounded-lg flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-sm">R</span>
            </div>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-green-500 to-blue-500 font-bold text-2xl tracking-wider">REPZ</span>
          </div>
          <div className="text-center sm:text-right">
            <p className="text-sm text-slate-600 font-medium">Last updated</p>
            <p className="text-xs text-slate-500">February 4, 2025</p>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 lg:py-12">
        <Card className="shadow-2xl border-0 bg-white/95 backdrop-blur-sm">
          <CardHeader className="bg-gradient-to-r from-green-50 via-blue-50 to-green-50 border-b border-gray-200">
            <div className="text-center py-8">
              <CardTitle className="text-4xl md:text-5xl font-bold text-slate-900 mb-4 flex items-center justify-center gap-4">
                <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-blue-500 rounded-xl flex items-center justify-center shadow-lg">
                  <Stethoscope className="h-6 w-6 text-white" />
                </div>
                Health Disclaimer
              </CardTitle>
              <p className="text-slate-600 text-lg max-w-2xl mx-auto leading-relaxed">
                Important health and medical information for safe fitness participation with REPZ
              </p>
              <div className="flex items-center justify-center gap-2 mt-4 text-sm text-slate-500">
                <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                <span>Medical Safety Required</span>
              </div>
            </div>
          </CardHeader>
          
          <CardContent className="prose prose-slate max-w-none p-8">
            
            {/* Critical Health Warning */}
            <div className="bg-red-100 border-2 border-red-300 rounded-lg p-6 mb-8">
              <div className="flex items-start gap-4">
                <Heart className="h-8 w-8 text-red-600 mt-1 flex-shrink-0" />
                <div>
                  <h2 className="text-2xl font-bold text-red-900 mb-3">🚨 CRITICAL HEALTH WARNING 🚨</h2>
                  <p className="text-red-800 font-bold text-lg mb-3">
                    THIS IS NOT MEDICAL ADVICE
                  </p>
                  <p className="text-red-800 font-semibold mb-3">
                    REPZ services are for general fitness and wellness purposes only. Our content, programs, and advice are not intended to:
                  </p>
                  <ul className="list-disc pl-6 space-y-1 text-red-800 font-medium">
                    <li>Diagnose, treat, cure, or prevent any disease or medical condition</li>
                    <li>Replace professional medical advice, diagnosis, or treatment</li>
                    <li>Serve as a substitute for consultations with qualified healthcare providers</li>
                    <li>Address specific medical conditions or health problems</li>
                  </ul>
                  <p className="text-red-800 font-bold mt-4 text-lg">
                    ALWAYS CONSULT YOUR DOCTOR BEFORE STARTING ANY EXERCISE PROGRAM
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-8">
              
              {/* Medical Clearance Required */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">1. Medical Clearance Requirements</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  Before beginning any exercise program through REPZ, you MUST obtain clearance from a qualified healthcare provider, especially if you have any of the following conditions:
                </p>
                
                <h3 className="text-xl font-semibold text-slate-800 mb-3">Cardiovascular Conditions:</h3>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>Heart disease, heart attack, or stroke history</li>
                  <li>High blood pressure (hypertension)</li>
                  <li>High cholesterol or triglycerides</li>
                  <li>Irregular heartbeat or heart palpitations</li>
                  <li>Chest pain or shortness of breath</li>
                  <li>Family history of sudden cardiac death</li>
                </ul>

                <h3 className="text-xl font-semibold text-slate-800 mb-3">Metabolic and Endocrine Conditions:</h3>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>Diabetes (Type 1 or Type 2)</li>
                  <li>Thyroid disorders</li>
                  <li>Metabolic syndrome</li>
                  <li>Kidney or liver disease</li>
                  <li>Eating disorders</li>
                </ul>

                <h3 className="text-xl font-semibold text-slate-800 mb-3">Musculoskeletal Conditions:</h3>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>Joint problems or arthritis</li>
                  <li>Back or neck injuries</li>
                  <li>Bone density issues (osteoporosis)</li>
                  <li>Recent surgeries or injuries</li>
                  <li>Chronic pain conditions</li>
                </ul>

                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-5 w-5 text-yellow-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-yellow-900">Special Populations</h4>
                      <p className="text-yellow-800 text-sm mt-1">
                        Pregnant women, individuals over 40, those with chronic conditions, and anyone taking medications require special medical clearance before exercising.
                      </p>
                    </div>
                  </div>
                </div>
              </section>

              {/* Coach Qualifications and Limitations */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">2. Coach Qualifications and Scope of Practice</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  REPZ fitness coaches are qualified fitness professionals, but they are NOT medical doctors, physical therapists, or licensed healthcare providers. Our coaches:
                </p>
                
                <h3 className="text-xl font-semibold text-slate-800 mb-3">CAN Provide:</h3>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>General fitness guidance and exercise programming</li>
                  <li>Motivation and accountability support</li>
                  <li>Basic nutrition education (not medical nutrition therapy)</li>
                  <li>Lifestyle and wellness coaching</li>
                  <li>Exercise form and technique instruction</li>
                </ul>

                <h3 className="text-xl font-semibold text-slate-800 mb-3">CANNOT Provide:</h3>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>Medical diagnosis or treatment</li>
                  <li>Prescription or medication advice</li>
                  <li>Treatment for injuries or medical conditions</li>
                  <li>Medical nutrition therapy for health conditions</li>
                  <li>Rehabilitation services</li>
                  <li>Emergency medical care</li>
                </ul>
              </section>

              {/* Warning Signs to Stop Exercise */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">3. Warning Signs - Stop Exercise Immediately</h2>
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
                  <div className="flex items-start gap-3">
                    <Heart className="h-6 w-6 text-red-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-red-900 text-lg">EMERGENCY - Call 911 Immediately</h4>
                      <p className="text-red-800 text-sm mt-1">
                        Stop exercising and seek immediate medical attention if you experience:
                      </p>
                    </div>
                  </div>
                </div>
                
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li><strong>Chest pain, pressure, or tightness</strong></li>
                  <li><strong>Severe shortness of breath</strong></li>
                  <li><strong>Dizziness, lightheadedness, or fainting</strong></li>
                  <li><strong>Irregular or rapid heartbeat</strong></li>
                  <li><strong>Nausea or vomiting during exercise</strong></li>
                  <li><strong>Pain radiating to arms, jaw, neck, or back</strong></li>
                  <li><strong>Confusion or disorientation</strong></li>
                  <li><strong>Severe weakness or fatigue</strong></li>
                  <li><strong>Vision changes or loss</strong></li>
                  <li><strong>Severe headache</strong></li>
                </ul>

                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mt-4">
                  <h4 className="font-semibold text-yellow-900">Contact Your Doctor if You Experience:</h4>
                  <ul className="list-disc pl-6 space-y-1 text-yellow-800 text-sm mt-2">
                    <li>Persistent joint or muscle pain</li>
                    <li>Unusual fatigue lasting more than 24 hours</li>
                    <li>Sleep disturbances or mood changes</li>
                    <li>Persistent soreness or swelling</li>
                    <li>Changes in appetite or weight</li>
                  </ul>
                </div>
              </section>

              {/* Medication and Supplement Disclaimers */}
              <section>
                <h2 className="text-2xl font-semibent text-slate-900 mb-4">4. Medication and Supplement Safety</h2>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                  <div className="flex items-start gap-3">
                    <Pill className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-blue-900">Medication Interactions</h4>
                      <p className="text-blue-800 text-sm mt-1">
                        Exercise can affect how medications work in your body. Always consult your healthcare provider about exercise if you take any medications.
                      </p>
                    </div>
                  </div>
                </div>
                
                <h3 className="text-xl font-semibold text-slate-800 mb-3">Important Considerations:</h3>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>Blood pressure medications may affect heart rate response to exercise</li>
                  <li>Diabetes medications may require adjustment with increased activity</li>
                  <li>Blood thinners may increase bleeding risk with certain activities</li>
                  <li>Some medications may cause dehydration or heat sensitivity</li>
                </ul>

                <h3 className="text-xl font-semibold text-slate-800 mb-3">Supplement Disclaimers:</h3>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>REPZ does not prescribe or recommend specific supplements</li>
                  <li>Dietary supplements are not regulated like medications</li>
                  <li>Supplements may interact with medications or medical conditions</li>
                  <li>Always consult your healthcare provider before taking supplements</li>
                  <li>Pregnant or nursing women should avoid most supplements</li>
                </ul>
              </section>

              {/* Nutrition Disclaimers */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">5. Nutrition Guidance Disclaimers</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  REPZ provides general nutrition education and meal planning guidance. This is NOT medical nutrition therapy and should not be used to treat medical conditions.
                </p>
                
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
                  <h4 className="font-semibold text-yellow-900">Consult a Registered Dietitian for:</h4>
                  <ul className="list-disc pl-6 space-y-1 text-yellow-800 text-sm mt-2">
                    <li>Diabetes or metabolic disorders</li>
                    <li>Food allergies or intolerances</li>
                    <li>Eating disorders or disordered eating</li>
                    <li>Kidney, liver, or heart disease</li>
                    <li>Pregnancy or breastfeeding nutrition needs</li>
                  </ul>
                </div>

                <p className="text-slate-700 leading-relaxed">
                  Individual nutritional needs vary greatly based on age, gender, activity level, medical history, and other factors. Our general recommendations may not be appropriate for everyone.
                </p>
              </section>

              {/* Results Disclaimers */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">6. Individual Results and Expectations</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  Fitness and health outcomes vary significantly between individuals. Factors affecting results include:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>Genetics and family history</li>
                  <li>Age, gender, and initial fitness level</li>
                  <li>Consistency with program adherence</li>
                  <li>Nutrition and lifestyle factors</li>
                  <li>Medical conditions and medications</li>
                  <li>Sleep quality and stress levels</li>
                </ul>
                
                <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
                  <p className="text-slate-700 font-semibold">
                    No guarantee is made regarding specific results, timeline, or outcomes. Testimonials and success stories represent individual experiences and may not be typical.
                  </p>
                </div>
              </section>

              {/* Pregnancy and Special Populations */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">7. Special Population Warnings</h2>
                
                <h3 className="text-xl font-semibold text-slate-800 mb-3">Pregnancy and Postpartum:</h3>
                <div className="bg-pink-50 border border-pink-200 rounded-lg p-4 mb-4">
                  <p className="text-pink-800 font-semibold">
                    Pregnant and postpartum women require specialized exercise programming and medical supervision. Always obtain clearance from your obstetrician before exercising.
                  </p>
                </div>

                <h3 className="text-xl font-semibold text-slate-800 mb-3">Youth (Under 18):</h3>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>Growing bodies have different exercise needs and limitations</li>
                  <li>Parental consent and supervision required</li>
                  <li>Focus should be on fun, movement, and skill development</li>
                  <li>Avoid intensive training that may harm developing joints</li>
                </ul>

                <h3 className="text-xl font-semibold text-slate-800 mb-3">Older Adults (65+):</h3>
                <ul className="list-disc pl-6 space-y-2 text-slate-700">
                  <li>Higher risk of cardiovascular events during exercise</li>
                  <li>Greater risk of falls and balance issues</li>
                  <li>May have multiple chronic conditions requiring medical oversight</li>
                  <li>Medications may affect exercise response</li>
                </ul>
              </section>

              {/* Emergency Procedures */}
              <section className="bg-red-50 border border-red-200 rounded-lg p-6">
                <h2 className="text-2xl font-semibold text-red-900 mb-4">Emergency Medical Information</h2>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold text-red-900 mb-2">If You Experience a Medical Emergency:</h4>
                    <ol className="list-decimal pl-6 space-y-1 text-red-800">
                      <li className="font-semibold">STOP exercising immediately</li>
                      <li className="font-semibold">Call 911 or emergency services</li>
                      <li>Do not drive yourself to the hospital</li>
                      <li>Follow emergency operator instructions</li>
                      <li>Contact your emergency contact</li>
                    </ol>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-red-800">
                      <Phone className="h-5 w-5" />
                      <span className="font-semibold">Emergency Services: 911</span>
                    </div>
                    <div className="flex items-center gap-2 text-red-800">
                      <Phone className="h-5 w-5" />
                      <span>Poison Control: 1-800-222-1222</span>
                    </div>
                    <div className="flex items-center gap-2 text-red-800">
                      <Mail className="h-5 w-5" />
                      <span>REPZ Emergency: emergency@repz.com</span>
                    </div>
                  </div>
                </div>
              </section>

              {/* Legal Acknowledgment */}
              <section className="bg-slate-50 border border-slate-200 rounded-lg p-6">
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">Acknowledgment</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  By using REPZ services, you acknowledge that:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700">
                  <li>You have read and understood this health disclaimer</li>
                  <li>You understand the risks associated with exercise</li>
                  <li>You will obtain appropriate medical clearance before exercising</li>
                  <li>You will stop exercising if you experience any warning signs</li>
                  <li>You understand that REPZ services are not medical advice</li>
                  <li>You are responsible for your own health and safety</li>
                </ul>
              </section>

            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}