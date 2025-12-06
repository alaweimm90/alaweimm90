import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from "@/ui/atoms/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { ArrowLeft, Shield, AlertTriangle, Mail, Phone, Lock } from 'lucide-react';
import { RepzLogo } from '@/ui/organisms/RepzLogo';

export default function PrivacyPolicy() {
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
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-lg flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-sm">R</span>
            </div>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-indigo-500 font-bold text-2xl tracking-wider">REPZ</span>
          </div>
          <div className="text-center sm:text-right">
            <p className="text-sm text-slate-600 font-medium">Last updated</p>
            <p className="text-xs text-slate-500">February 4, 2025</p>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 lg:py-12">
        <Card className="shadow-2xl border-0 bg-white/95 backdrop-blur-sm">
          <CardHeader className="bg-gradient-to-r from-blue-50 via-indigo-50 to-blue-50 border-b border-gray-200">
            <div className="text-center py-8">
              <CardTitle className="text-4xl md:text-5xl font-bold text-slate-900 mb-4 flex items-center justify-center gap-4">
                <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-xl flex items-center justify-center shadow-lg">
                  <Shield className="h-6 w-6 text-white" />
                </div>
                Privacy Policy
              </CardTitle>
              <p className="text-slate-600 text-lg max-w-2xl mx-auto leading-relaxed">
                How we collect, use, and protect your personal information with industry-leading security
              </p>
              <div className="flex items-center justify-center gap-2 mt-4 text-sm text-slate-500">
                <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                <span>GDPR & CCPA Compliant</span>
              </div>
            </div>
          </CardHeader>
          
          <CardContent className="prose prose-slate max-w-none p-8">
            <div className="space-y-8">
              
              {/* Introduction */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">1. Introduction</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  REPZ ("we," "our," or "us") is committed to protecting your privacy and personal information. This Privacy Policy explains how we collect, use, disclose, and safeguard your information when you use our fitness coaching platform.
                </p>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <Lock className="h-5 w-5 text-blue-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-blue-900">Your Privacy Rights</h4>
                      <p className="text-blue-800 text-sm mt-1">
                        You have the right to know what personal information we collect, how it's used, and how to control your data.
                      </p>
                    </div>
                  </div>
                </div>
              </section>

              {/* Information We Collect */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">2. Information We Collect</h2>
                
                <h3 className="text-xl font-semibold text-slate-800 mb-3">2.1 Personal Information</h3>
                <p className="text-slate-700 leading-relaxed mb-4">
                  We collect information you provide directly to us, including:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>Name, email address, and contact information</li>
                  <li>Account credentials (username and password)</li>
                  <li>Profile information (age, gender, fitness goals)</li>
                  <li>Payment and billing information</li>
                  <li>Communication preferences</li>
                </ul>

                <h3 className="text-xl font-semibold text-slate-800 mb-3">2.2 Health and Fitness Data</h3>
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-5 w-5 text-yellow-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-yellow-900">Sensitive Health Information</h4>
                      <p className="text-yellow-800 text-sm mt-1">
                        We collect health-related data to provide personalized coaching. This information receives special protection under privacy laws.
                      </p>
                    </div>
                  </div>
                </div>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>Body measurements (weight, height, body composition)</li>
                  <li>Fitness assessments and progress tracking</li>
                  <li>Workout performance and exercise data</li>
                  <li>Nutrition information and dietary preferences</li>
                  <li>Sleep patterns and recovery metrics</li>
                  <li>Medical conditions and injury history (if provided)</li>
                </ul>

                <h3 className="text-xl font-semibold text-slate-800 mb-3">2.3 Usage and Technical Data</h3>
                <ul className="list-disc pl-6 space-y-2 text-slate-700">
                  <li>Device information (IP address, browser type, operating system)</li>
                  <li>Usage patterns and feature interactions</li>
                  <li>Performance data and error reports</li>
                  <li>Location data (if you enable location services)</li>
                  <li>Cookies and similar tracking technologies</li>
                </ul>
              </section>

              {/* How We Use Information */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">3. How We Use Your Information</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  We use your information to:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>Provide personalized fitness coaching and training programs</li>
                  <li>Track your progress and provide performance analytics</li>
                  <li>Process payments and manage your subscription</li>
                  <li>Communicate with you about your account and our services</li>
                  <li>Improve our platform and develop new features</li>
                  <li>Ensure platform security and prevent fraud</li>
                  <li>Comply with legal obligations</li>
                </ul>
                <p className="text-slate-700 leading-relaxed">
                  We do NOT use your health information for insurance purposes, employment decisions, or any discriminatory practices.
                </p>
              </section>

              {/* Information Sharing */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">4. Information Sharing and Disclosure</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  We do not sell, rent, or trade your personal information. We may share your information only in these limited circumstances:
                </p>
                
                <h3 className="text-xl font-semibold text-slate-800 mb-3">4.1 Service Providers</h3>
                <p className="text-slate-700 leading-relaxed mb-4">
                  We may share information with trusted third-party service providers who assist us in:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>Payment processing (Stripe)</li>
                  <li>Email communications</li>
                  <li>Data analytics and platform improvement</li>
                  <li>Customer support services</li>
                </ul>

                <h3 className="text-xl font-semibold text-slate-800 mb-3">4.2 Legal Requirements</h3>
                <p className="text-slate-700 leading-relaxed mb-4">
                  We may disclose information when required by law or to:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>Comply with legal process or government requests</li>
                  <li>Protect our rights, property, or safety</li>
                  <li>Protect the rights and safety of our users</li>
                  <li>Investigate potential violations of our terms</li>
                </ul>

                <h3 className="text-xl font-semibold text-slate-800 mb-3">4.3 Emergency Situations</h3>
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-5 w-5 text-red-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-red-900">Emergency Disclosure</h4>
                      <p className="text-red-800 text-sm mt-1">
                        In life-threatening emergencies, we may share relevant health information with emergency responders or medical professionals.
                      </p>
                    </div>
                  </div>
                </div>
              </section>

              {/* Data Security */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">5. Data Security and Protection</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  We implement industry-standard security measures to protect your information:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>Encryption of data in transit and at rest</li>
                  <li>Secure server infrastructure with regular security updates</li>
                  <li>Access controls and authentication requirements</li>
                  <li>Regular security audits and vulnerability assessments</li>
                  <li>Staff training on data protection and privacy</li>
                </ul>
                <p className="text-slate-700 leading-relaxed">
                  However, no internet transmission is completely secure. We cannot guarantee absolute security of your information.
                </p>
              </section>

              {/* Data Retention */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">6. Data Retention</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  We retain your information for as long as necessary to:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>Provide our services and maintain your account</li>
                  <li>Comply with legal obligations</li>
                  <li>Resolve disputes and enforce agreements</li>
                  <li>Improve our services (in aggregated, de-identified form)</li>
                </ul>
                <p className="text-slate-700 leading-relaxed">
                  When you delete your account, we will delete or anonymize your personal information within 30 days, except where retention is required by law.
                </p>
              </section>

              {/* Your Rights */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">7. Your Privacy Rights</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  Depending on your location, you may have the following rights:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li><strong>Access:</strong> Request a copy of your personal information</li>
                  <li><strong>Correction:</strong> Update or correct inaccurate information</li>
                  <li><strong>Deletion:</strong> Request deletion of your personal information</li>
                  <li><strong>Portability:</strong> Receive your data in a machine-readable format</li>
                  <li><strong>Restriction:</strong> Limit how we process your information</li>
                  <li><strong>Objection:</strong> Object to certain types of processing</li>
                  <li><strong>Withdraw Consent:</strong> Withdraw previously given consent</li>
                </ul>
                <p className="text-slate-700 leading-relaxed">
                  To exercise these rights, contact us using the information provided below.
                </p>
              </section>

              {/* International Transfers */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">8. International Data Transfers</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  Your information may be transferred to and processed in countries other than your own. We ensure adequate protection through:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700">
                  <li>Standard contractual clauses approved by regulatory authorities</li>
                  <li>Adequacy decisions by relevant privacy authorities</li>
                  <li>Other appropriate safeguards as required by law</li>
                </ul>
              </section>

              {/* Children's Privacy */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">9. Children's Privacy</h2>
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-5 w-5 text-yellow-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-yellow-900">Age Restriction</h4>
                      <p className="text-yellow-800 text-sm mt-1">
                        REPZ is not intended for children under 16. We do not knowingly collect personal information from children under 16.
                      </p>
                    </div>
                  </div>
                </div>
                <p className="text-slate-700 leading-relaxed">
                  If we discover that we have collected information from a child under 16, we will delete that information immediately. If you believe we have collected information from a child, please contact us.
                </p>
              </section>

              {/* Changes to Policy */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">10. Changes to This Privacy Policy</h2>
                <p className="text-slate-700 leading-relaxed">
                  We may update this Privacy Policy from time to time. Material changes will be communicated via email or platform notification at least 30 days before taking effect. Your continued use of REPZ constitutes acceptance of the updated policy.
                </p>
              </section>

              {/* Contact Information */}
              <section className="bg-slate-50 border border-slate-200 rounded-lg p-6">
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">Contact Us</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  For questions about this Privacy Policy or to exercise your privacy rights, contact us:
                </p>
                <div className="space-y-2 ml-4">
                  <div className="text-slate-700">Mon-Fri 6AM-8PM PST</div>
                  <div className="flex items-center gap-2 text-slate-700">
                    <Phone className="h-4 w-4" />
                    <span>+1 (415) 992-9792</span>
                  </div>
                  <div className="flex items-center gap-2 text-slate-700">
                    <Mail className="h-4 w-4" />
                    <span>contact@repz.com</span>
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